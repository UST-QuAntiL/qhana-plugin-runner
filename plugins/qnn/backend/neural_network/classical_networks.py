import torch
from torch import nn
from plugins.qnn.schemas import WeightInitEnum


class FeedForwardNetwork(nn.Module):
    """
    Torch module implementing the classical net.
    """

    def __init__(self, n_features, depth, weight_init):
        """
        Initialize network with preprocessing, classical and postprocessing layers

        n_features: number of features per layer
        depth: number of layers
        weight_init: type of (random) initialization of the models weights (WeightInitEnum)
        """

        super().__init__()

        self.pre_net = nn.Linear(2, n_features)

        self.relu = nn.ReLU()
        self.classical_net = nn.ModuleList(
            [nn.Linear(n_features, n_features) for i in range(depth)]
        )

        self.post_net = nn.Linear(n_features, 2)

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

        # preprocessing layer
        out = self.pre_net(input_features)
        # out = torch.tanh(out)

        # classical net
        for i, layer in enumerate(self.classical_net):
            out = self.relu(layer(out))

        c_out = torch.tanh(out)

        # postprocessing layer
        return self.post_net(c_out)
