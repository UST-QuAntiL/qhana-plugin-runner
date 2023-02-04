from enum import Enum


class WeightInitEnum(Enum):
    standard_normal = "standard normal"
    uniform = "uniform"
    zero = "zero"


class NeuralNetworkEnum(Enum):
    feed_forward_net = "Feed Forward Neural Network"
    dressed_quantum_net = "Dressed Quantum Neural Network"

    def get_neural_network(self, parameters):
        if self == NeuralNetworkEnum.feed_forward_net:
            from .classical_networks import FeedForwardNetwork

            return FeedForwardNetwork(**parameters)
        elif self == NeuralNetworkEnum.dressed_quantum_net:
            from .quantum_networks import DressedQuantumNet

            return DressedQuantumNet(**parameters)

    def needs_quantum_backend(self):
        if self == NeuralNetworkEnum.feed_forward_net:
            return False
        elif self == NeuralNetworkEnum.dressed_quantum_net:
            return True
