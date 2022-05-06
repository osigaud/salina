import sys
import os
import torch
import torch.nn as nn
import numpy as np

import gym
import my_gym

from omegaconf import OmegaConf
from salina.workspace import Workspace
from salina.agent import Agent
from salina.chrono import Chrono
from torch.distributions.normal import Normal
from salina.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from salina.agents.utils import Agents, TemporalAgent
from salina import instantiate_class, get_arguments, get_class


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


class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments without auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env


class ContinuousActionTunableVarianceAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
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
            action = dist.sample()  # valid actions are supposed to be in [-1,1] range
        else:
            action = mean  # valid actions are supposed to be in [-1,1] range
        logp_pi = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_pi)

    def predict_action(self, obs, stochastic):
        mean = self.model(obs)
        dist = Normal(mean, self.soft_plus(self.std_param))
        if stochastic:
            action = dist.sample()  # valid actions are supposed to be in [-1,1] range
        else:
            action = mean  # valid actions are supposed to be in [-1,1] range
        return action


def make_gym_env(max_episode_steps, env_name):
    return gym.make(env_name)


# Create the A2C Agent
def create_a2c_agent(eval_env_agent, filename):

    observation_size, action_dim = eval_env_agent.get_obs_and_actions_sizes()
    action_agent = ContinuousActionTunableVarianceAgent(observation_size, None, action_dim)
    action_agent.load_model(filename)
    param_agent = action_agent
    ev_agent = Agents(eval_env_agent, action_agent)

    # Get an agent that is executed on a complete workspace
    eval_agent = TemporalAgent(ev_agent)
    return eval_agent, param_agent


def evaluate_agent(cfg, filename):

    # 2) Create the environment agent
    eval_env_agent = NoAutoResetEnvAgent(cfg, n_envs=cfg.algorithm.nb_evals)

    # 3) Create the A2C Agent
    eval_agent, param_agent = create_a2c_agent(eval_env_agent, filename)

    eval_workspace = Workspace()  # Used for evaluation
    eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)
    rewards = eval_workspace["env/cumulated_reward"][-1]
    mean = rewards.mean()
    print(f"reward: {mean}")
    return mean


params = {
    "algorithm": {
        "seed": 4,
        "nb_evals": 900,
    },
    "gym_env": {"classname": "__main__.make_gym_env",
                "env_name": "CartPoleContinuous-v1"},
}

if __name__ == "__main__":
    chrono = Chrono()
    sys.path.append(os.getcwd())
    config = OmegaConf.create(params)
    folder = "./data/policies"
    listdir = os.listdir(folder)
    for policy_file in listdir:
        val = evaluate_agent(config, folder + "/" + policy_file)
        print(f"{policy_file}: {val}")
    chrono.stop()

