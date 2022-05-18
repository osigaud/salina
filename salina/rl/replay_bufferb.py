#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from salina.workspace import Workspace
import copy


class ReplayBuffer:
    def __init__(self, max_size, device=torch.device("cpu")):
        self.max_size = max_size
        self.variables = None
        self.position = 0
        self.is_full = False
        self.device = device

    def put(self, workspace, time_size=None, padding=None):
        self.variables = workspace.variables

        all_tensors = {k: workspace.get_full(k).detach().to(self.device) for k in workspace.keys()}
        batch_size = None
        arange = None
        indexes = None
        for k, v in all_tensors.items():
            if batch_size is None:
                batch_size = v.size()[1]
            batch_size = min(self.position + batch_size, self.max_size)
            batch_size = batch_size - self.position
            if indexes is None:
                indexes = torch.arange(batch_size) + self.position
                arange = torch.arange(batch_size)
            indexes = indexes.to(v.device)
            arange = arange.to(v.device)
            self.variables[k][indexes] = v.detach().transpose(0, 1)

        self.position = self.position + batch_size
        if self.position >= self.max_size:
            self.position = 0
            self.is_full = True

    def size(self):
        if self.is_full:
            return self.max_size
        else:
            return self.position

    def get_shuffled(self, batch_size):
        who = torch.randint(low=0, high=self.size(), size=(batch_size,), device=self.device)
        workspace = Workspace()
        for k in self.variables:
            workspace.set_full(k, self.variables[k][who].transpose(0, 1))

        return workspace

    def to(self, device):
        n_vars = {k: v.to(device) for k, v in self.variables.items()}
        self.variables = n_vars
