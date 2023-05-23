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

from torch import nn
from torch import Tensor
from . import WeightInitEnum
from typing import List


def create_fully_connected_net(
    input_size: int, hidden_layers: List[int], output_size: int
) -> nn.Sequential:
    """
    Returns a nn.Sequential network. It contains linear layers with the given layer sizes and ReLU as an activation
    function after each layer. The output layer has no activation function
    :param input_size: integer determining the input layer size
    :param hidden_layers: list of integers determining the hidden layer sizes
    :param output_size: list of integers determining the output layer size
    """
    net = nn.Sequential()
    if len(hidden_layers) > 0:
        net.add_module("input_layer", nn.Linear(input_size, hidden_layers[0]))

        for idx, layer_size in enumerate(hidden_layers[:-1]):
            net.add_module(f"act_func_{idx}", nn.ReLU())
            net.add_module(
                f"hidden_layer_{idx}", nn.Linear(layer_size, hidden_layers[idx + 1])
            )

        net.add_module(f"act_func_{len(hidden_layers)}", nn.ReLU())
        net.add_module("output_layer", nn.Linear(hidden_layers[-1], output_size))
    else:
        net.add_module("output_layer", nn.Linear(input_size, output_size))

    return net


class FeedForwardNetwork(nn.Module):
    """
    Torch module implementing the classical net.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int],
        weight_init: WeightInitEnum,
        **kwargs,
    ):
        """
        Initialize network with preprocessing, classical and postprocessing layers

        n_features: number of features per layer
        depth: number of layers
        weight_init: type of (random) initialization of the models weights (WeightInitEnum)
        """

        super().__init__()

        self.net = create_fully_connected_net(input_size, hidden_layers, output_size)

        # weight initialization
        self.weight_init = weight_init
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            # init weights according to initialization type
            if self.weight_init == WeightInitEnum.standard_normal:
                module.weight.data.normal_(mean=0.0, std=1.0)
            elif self.weight_init == WeightInitEnum.uniform:
                module.weight.data.uniform_()
            elif self.weight_init == WeightInitEnum.zero:
                module.weight.data.zero_()
            else:
                raise NotImplementedError("unknown weight init method")

            # initialize bias
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_features: Tensor):
        """
        pass input features through classical layers
        """
        return self.net(input_features)
