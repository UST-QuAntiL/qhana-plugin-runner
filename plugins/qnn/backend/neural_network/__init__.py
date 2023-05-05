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

from enum import Enum
from torch.nn import Module


class WeightInitEnum(Enum):
    standard_normal = "standard normal"
    uniform = "uniform"
    zero = "zero"


class NeuralNetworkEnum(Enum):
    feed_forward_net = "Feed Forward Neural Network"
    dressed_quantum_net = "Dressed Quantum Neural Network"

    def get_neural_network(self, parameters: dict) -> Module:
        if self == NeuralNetworkEnum.feed_forward_net:
            from .classical_networks import FeedForwardNetwork

            return FeedForwardNetwork(**parameters)
        elif self == NeuralNetworkEnum.dressed_quantum_net:
            from .quantum_networks import DressedQuantumNet

            return DressedQuantumNet(**parameters)

    def needs_quantum_backend(self) -> bool:
        if self == NeuralNetworkEnum.feed_forward_net:
            return False
        elif self == NeuralNetworkEnum.dressed_quantum_net:
            return True
