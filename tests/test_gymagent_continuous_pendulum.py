import sys
import os

import gym
import my_gym
import time

from gym.wrappers import TimeLimit
from omegaconf import OmegaConf
from salina import instantiate_class, get_arguments, get_class
from salina.workspace import Workspace
from salina.agents import Agents, TemporalAgent, PrintAgent

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from salina.agent import Agent
from salina.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent

from salina.visu.visu_policies import plot_policy
from salina.visu.visu_critics import plot_critic


def get_transitions(workspace):
    """
    Takes in a workspace from salina:
    [(step1),(step2),(step3), ... ]
    return a workspace of transitions :
    [
        [step1,step2],
        [step2,step3]
        ...
    ]
    Filters every transitions [step_final,step_initial]
    """
    transitions = {}
    done = workspace["env/done"][:-1]
    for key in workspace.keys():
        array = workspace[key]

        # remove transitions (s_terminal -> s_initial)
        x = array[:-1][~done]
        x_next = array[1:][~done]
        transitions[key] = torch.stack([x, x_next])

    workspace = Workspace()
    for k, v in transitions.items():
        workspace.set_full(k, v)
    return workspace


def build_backbone(sizes, activation):
    layers = []
    for j in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation]
    return layers


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class ContinuousActionTunableVarianceAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, **kwargs):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.ReLU())
        init_variance = torch.randn(action_dim, 1)
        # print("init_variance:", init_variance)
        self.std_param = nn.parameter.Parameter(init_variance)
        self.soft_plus = torch.nn.Softplus()

    def forward(self, t, stochastic, **kwargs):
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)
        dist = Normal(mean, self.soft_plus(self.std_param))  # std must be positive
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = torch.tanh(dist.sample())  # valid actions are supposed to be in [-1,1] range
        else:
            action = torch.tanh(mean)  # valid actions are supposed to be in [-1,1] range
        logp_pi = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_pi)

    def predict_action(self, obs, stochastic):
        mean = self.model(obs)
        dist = Normal(mean, self.soft_plus(self.std_param))
        if stochastic:
            action = torch.tanh(dist.sample())  # valid actions are supposed to be in [-1,1] range
        else:
            action = torch.tanh(mean)  # valid actions are supposed to be in [-1,1] range
        return action


class ContinuousActionStateDependentVarianceAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, **kwargs):
        super().__init__()
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_backbone(backbone_dim, activation=nn.ReLU())
        self.last_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.mean_layer = nn.Tanh()
        # std must be positive
        self.std_layer = nn.Softplus()
        self.backbone = nn.Sequential(*self.layers)

    def forward(self, t, stochastic, **kwargs):
        obs = self.get(("env/env_obs", t))
        backbone_output = self.backbone(obs)
        last = self.last_layer(backbone_output)
        # print(last)
        mean = self.mean_layer(last)
        assert not torch.any(torch.isnan(mean)), "Nan Here"
        dist = Normal(mean, self.std_layer(last))
        self.set(("entropy", t), dist.entropy())
        if stochastic:
            action = torch.tanh(dist.sample())  # valid actions are supposed to be in [-1,1] range
        else:
            action = torch.tanh(mean)  # valid actions are supposed to be in [-1,1] range
        logp_pi = dist.log_prob(action).sum(axis=-1)
        # print(f"action: {action}")
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_pi)

    def predict_action(self, obs, stochastic):
        backbone_output = self.backbone(obs)
        last = self.last_layer(backbone_output)
        mean = self.mean_layer(last)
        dist = Normal(mean, self.std_layer(last))
        if stochastic:
            action = torch.tanh(dist.sample())  # valid actions are supposed to be in [-1,1] range
        else:
            action = torch.tanh(mean)  # valid actions are supposed to be in [-1,1] range
        return action

class VAgent(Agent):
    def __init__(self, state_dim, hidden_layers):
        super().__init__()
        self.model = build_mlp([state_dim] + list(hidden_layers) + [1], activation=nn.ReLU())

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)


class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, epoch):
        self.logger.add_scalar(log_string, loss.item(), epoch)

    # Log losses
    def log_losses(self, epoch, critic_loss, entropy_loss, a2c_loss):
        self.add_log("critic_loss", critic_loss, epoch)
        self.add_log("entropy_loss", entropy_loss, epoch)
        self.add_log("a2c_loss", a2c_loss, epoch)


class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, max_episode_steps, n_envs):
        super().__init__(max_episode_steps, get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments without auto-reset
    def __init__(self, cfg, max_episode_steps, n_envs):
        super().__init__(max_episode_steps, get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env


# Create the A2C Agent
def create_a2c_agent(cfg, train_env_agent, eval_env_agent):
    observation_size, n_actions = train_env_agent.get_obs_and_actions_sizes()
    action_agent = ContinuousActionTunableVarianceAgent(observation_size, cfg.algorithm.architecture.hidden_size, n_actions)
    tr_agent = Agents(train_env_agent, action_agent)
    ev_agent = Agents(eval_env_agent, action_agent)

    critic_agent = VAgent(observation_size, cfg.algorithm.architecture.hidden_size)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, eval_agent, critic_agent


def make_gym_env(max_episode_steps, env_name):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, action_agent, critic_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = nn.Sequential(action_agent, critic_agent).parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, critic):
    # Compute temporal difference
    target = reward[:-1] + cfg.algorithm.discount_factor * critic[1:].detach() * (must_bootstrap.float())
    td = target - critic[:-1]

    # Compute critic loss
    td_error = td ** 2
    critic_loss = td_error.mean()
    return critic_loss, td


def compute_actor_loss_continuous(action_logp, td):
    a2c_loss = action_logp[:-1] * td.detach()
    return a2c_loss.mean()


def run_a2c(cfg, max_grad_norm=0.5):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agent
    train_env_agent = AutoResetEnvAgent(cfg, max_episode_steps=cfg.gym_env.max_episode_steps,
                                        n_envs=cfg.algorithm.n_envs)
    eval_env_agent = NoAutoResetEnvAgent(cfg, max_episode_steps=cfg.gym_env.max_episode_steps,
                                         n_envs=cfg.algorithm.nb_evals)

    # 3) Create the A2C Agent
    a2c_agent, eval_agent, critic_agent = create_a2c_agent(cfg, train_env_agent, eval_env_agent)

    # 4) Create the temporal critic agent to compute critic values over the workspace
    tcritic_agent = TemporalAgent(critic_agent)

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    train_workspace = Workspace()  # Used for training

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, a2c_agent, critic_agent)
    nb_steps = 0
    tmp_steps = 0

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            a2c_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps - 1, stochastic=True)
        else:
            a2c_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True)

        # Compute the critic value over the whole workspace
        tcritic_agent(train_workspace, n_steps=cfg.algorithm.n_steps)
        nb_steps += cfg.algorithm.n_steps * cfg.algorithm.n_envs

        transition_workspace = get_transitions(train_workspace)

        critic, done, reward, action, action_logp, truncated = transition_workspace[
                "critic", "env/done", "env/reward", "action", "action_logprobs", "env/truncated"]

        must_bootstrap = torch.logical_or(~done[1], truncated[1])

        critic_loss, td = compute_critic_loss(cfg, reward, must_bootstrap, critic)

        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()

        # Compute entropy loss
        entropy_loss = torch.mean(train_workspace["entropy"])

        # Store the losses for tensorboard display
        logger.log_losses(nb_steps, critic_loss, entropy_loss, a2c_loss)

        # Compute the total loss
        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a2c_agent.parameters(), max_grad_norm)
        optimizer.step()

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"epoch: {epoch}, reward: {mean }")

            if cfg.save_best and mean > best_reward:
                best_reward = mean
                filename = "./tmp/a2c" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)
                policy = eval_agent.agent.agents[1]
                critic = critic_agent
                plot_policy(policy, eval_env_agent, "Pendulum-v1", "./tmp/", best_reward, stochastic=False)
                plot_critic(critic, eval_env_agent, "Pendulum-v1", "./tmp/", best_reward)

params = {
    "save_best": True,
    "logger": {"classname": "salina.logger.TFLogger",
               "log_dir": "./tmp/" + str(time.time()),
               "verbose": False,
               # "cache_size": 10000,
               "every_n_seconds": 10},
    "algorithm": {
        "seed": 5,
        "n_envs": 1,
        "n_steps": 8,
        "eval_interval": 100,
        "nb_evals": 10,
        "gae": 0.8,
        "max_epochs": 50000,
        "discount_factor": 0.95,
        "entropy_coef": 2.55e-7,
        "critic_coef": 0.4,
        "a2c_coef": 1,
        "architecture": {"hidden_size": [64, 64]},
    },
    "gym_env": {"classname": "__main__.make_gym_env",
                "env_name": "Pendulum-v1",
                "max_episode_steps": 200},
    "optimizer": {"classname": "torch.optim.RMSprop",
                  "lr": 0.004},
}


if __name__ == "__main__":
    # with autograd.detect_anomaly():
    sys.path.append(os.getcwd())
    config = OmegaConf.create(params)
    torch.manual_seed(config.algorithm.seed)
    run_a2c(config)
