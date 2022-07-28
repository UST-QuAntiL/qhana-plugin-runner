# Copyright 2022 QHAna plugin runner contributors.
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
from typing import List

import numpy as np
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, number_of_input_values: int, number_of_hidden_units: int):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(number_of_input_values, number_of_hidden_units),
            nn.ReLU(),
            nn.Linear(number_of_hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

    def get_loss(self, input_data: torch.Tensor, target_data: torch.Tensor):
        with torch.no_grad():
            output = self(input_data)

            loss = torch.mean((output - target_data) * (output - target_data))

        return loss.item()

    def get_param_list(self) -> List[float]:
        params = list(self.parameters())
        param_list = []

        for param in params:
            param_list.extend(param.data.detach().flatten().tolist())

        return param_list

    def set_param_list(self, params: List[float]):
        index = 0

        for param in self.parameters():
            number_of_elements = param.data.numel()
            param.data = torch.tensor(
                params[index : index + number_of_elements], dtype=torch.float32
            ).resize_as_(param.data)
            index += number_of_elements

    def set_param_get_loss(
        self, x: np.ndarray, input_data: torch.Tensor, target_data: torch.Tensor
    ):
        self.set_param_list(list(x))

        return self.get_loss(input_data, target_data)

    def get_number_of_parameters(self):
        return len(self.get_param_list())
