from enum import Enum


class WeightInitEnum(Enum):
    standard_normal = "standard normal"
    uniform = "uniform"
    zero = "zero"


class NeuralNetworkEnum(Enum):
    feed_forward_net = "Feed Forward Neural Network"
    dressed_quantum_net = "Dressed Quantum Neural Network"

    def get_neural_network(self, parameters: dict):
        if self == NeuralNetworkEnum.feed_forward_net:
            from .classical_networks import FeedForwardNetwork

            return FeedForwardNetwork(
                parameters["input_size"],
                parameters["output_size"],
                parameters["n_qubits"],
                parameters["depth"],
                parameters["weight_init"],
            )
        elif self == NeuralNetworkEnum.dressed_quantum_net:
            from .quantum_networks import DressedQuantumNet

            return DressedQuantumNet(
                parameters["input_size"],
                parameters["output_size"],
                parameters["n_qubits"],
                parameters["quantum_device"],
                parameters["depth"],
                parameters["weight_init"],
                parameters["preprocess_layers"],
                parameters["postprocess_layers"],
                parameters["single_q_params"],
            )

    def needs_quantum_backend(self):
        if self == NeuralNetworkEnum.feed_forward_net:
            return False
        elif self == NeuralNetworkEnum.dressed_quantum_net:
            return True
