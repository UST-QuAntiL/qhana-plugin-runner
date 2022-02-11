import time
from enum import Enum
from itertools import chain
from math import pi
from os import PathLike
from typing import Iterator, Dict, List, Union, BinaryIO, IO

import torch
import pennylane as qml
from pyquil import get_qc
from pyquil.api import QuantumComputer
from torch import Tensor

import plugins.hybrid_ae_pkg.backend.quantum.pytorch_backend.pennylane_backend.qlayer as pennylane_backend
import plugins.hybrid_ae_pkg.backend.quantum.pytorch_backend.pyquil_backend.layer as pyquil_backend
from plugins.hybrid_ae_pkg.backend.quantum.pytorch_backend.pyquil_backend.circuit_logging import (
    CircuitLogger,
)


class Backend(Enum):
    pennylane = 0
    pyquil = 1


class HybridAutoencoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        q_num: int,
        embedding_size: int,
        qnn_name: str,
        backend: Backend,
        dev: Union[qml.Device, QuantumComputer],
    ):
        super(HybridAutoencoder, self).__init__()

        self.q_num = q_num
        self.embedding_size = embedding_size

        self.fc1 = torch.nn.Linear(input_size, q_num)

        if backend == Backend.pennylane:
            self.q_layer1 = pennylane_backend.create_qlayer(
                pennylane_backend.qnn_constructors[qnn_name], q_num, dev
            )
            self.q_layer2 = pennylane_backend.create_qlayer(
                pennylane_backend.qnn_constructors[qnn_name], q_num, dev
            )
        elif backend == Backend.pyquil:
            logger = CircuitLogger()
            self.q_layer1 = pyquil_backend.create_qlayer(
                pyquil_backend.qnn_constructors[qnn_name], q_num, 1, dev, logger
            )
            self.q_layer2 = pyquil_backend.create_qlayer(
                pyquil_backend.qnn_constructors[qnn_name], q_num, 1, dev, logger
            )

        self.fc2 = torch.nn.Linear(q_num, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the output of the model.

        :param x: Input. The range of the values can be anything, but should be appropriately scaled.
        :return: Output of the model. The range of the values can be anything.
        """
        embedding = self.embed(x)
        reconstruction = self.reconstruct(embedding)

        return reconstruction

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        x = torch.sigmoid(self.fc1(x))
        x = (
            x * pi
        )  # scale with pi because input to the quantum layer should be in the range [0, pi]
        embedding = self.q_layer1(x)  # output in the range [-1, 1]

        # scaling the values to be in the range [0, pi]
        embedding = (embedding / 2.0 + 0.5) * pi
        # bottleneck
        embedding = embedding[:, 0 : self.embedding_size]

        return embedding

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        # padding with zeros to match the input size of the quantum layer
        x = torch.cat(
            (x, torch.zeros((x.shape[0], self.q_num - self.embedding_size))), dim=1
        )
        # decoder
        x = self.q_layer2(x)
        reconstruction = self.fc2(x)

        return reconstruction

    def get_classical_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(self.fc1.parameters(), self.fc2.parameters())

    def get_quantum_parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        return chain(self.q_layer1.parameters(), self.q_layer2.parameters())

    def save_model_parameters(self, file_path: Union[str, PathLike, BinaryIO, IO[bytes]]):
        torch.save(self.state_dict(), file_path)

    def load_model_parameters(self, file_path: Union[str, PathLike, BinaryIO, IO[bytes]]):
        self.load_state_dict(torch.load(file_path))

    def get_model_parameters_as_dict(self) -> Dict[str, List]:
        param_dict: Dict[str, List] = {}

        for k, v in self.state_dict().items():
            param_dict[k] = v.tolist()

        return param_dict


def _hybrid_autoencoder_example():
    model = HybridAutoencoder(4, 2, 1, "QNN3", Backend.pyquil, get_qc("4q-qvm"))

    training_input = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]])

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    model.train()

    for i in range(100):
        time1 = time.time()
        pred: Tensor = model(training_input)
        loss = loss_fn(pred, training_input)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time2 = time.time()

        print(f"loss: {loss.item():>7f} output: {pred} time: {time2 - time1}")


if __name__ == "__main__":
    _hybrid_autoencoder_example()
