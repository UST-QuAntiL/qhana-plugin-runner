import torch
from torch import nn
from plugins.qnn.schemas import WeightInitEnum
from typing import List


def create_fully_connected_net(
    input_size: int, hidden_layers: List[int], output_size: int
) -> nn.Sequential:
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

    def __init__(self, input_size, output_size, hidden_layers, weight_init):
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # init weights according to initialization type
            if self.weight_init == WeightInitEnum.standard_normal:
                module.weight.data.normal_(mean=0.0, std=1.0)
            elif self.weight_init == WeightInitEnum.uniform:
                module.weight.data.uniform_()
            elif self.weight_init == WeightInitEnum.zero:  # TODO plot is completely blue?
                module.weight.data.zero_()
            else:
                raise NotImplementedError("unknown weight init method")

            # initialize bias
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_features):
        """
        pass input features through classical layers
        """
        return self.net(input_features)
