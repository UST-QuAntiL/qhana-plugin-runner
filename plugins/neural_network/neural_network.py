# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
from torch import nn


class NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(NN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

    def set_weights(self, params: np.ndarray):
        index = 0
        for param in self.parameters():
            number_of_elements = param.data.numel()
            param.data = torch.tensor(
                params[index : index + number_of_elements], dtype=torch.float32
            ).view_as(param.data)
            index += number_of_elements

    def get_loss(self, input_data: torch.Tensor, target_data: torch.Tensor):
        # Compute the output of the network
        output = self(input_data)

        # Compute the mean squared error loss
        loss = torch.mean((output - target_data) * (output - target_data))

        return loss.item()

    def get_gradient(self, input_data: torch.Tensor, target_data: torch.Tensor):
        # Zero the existing gradients
        self.zero_grad()

        output = self(input_data)
        loss = torch.mean((output - target_data) * (output - target_data))
        loss.backward()
        grads = np.concatenate(
            [p.grad.data.cpu().numpy().flatten() for p in self.parameters()]
        ).astype(np.float64)

        return grads

    def get_loss_and_gradient(self, input_data: torch.Tensor, target_data: torch.Tensor):
        # Zero the existing gradients
        self.zero_grad()

        output = self(input_data)
        loss = torch.mean((output - target_data) * (output - target_data))
        loss.backward()
        grads = np.concatenate(
            [p.grad.data.cpu().numpy().flatten() for p in self.parameters()]
        ).astype(np.float64)

        return loss.item(), grads
