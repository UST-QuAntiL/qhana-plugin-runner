import math
import os
import random
from typing import List, Optional

import numpy as np
import torch
from .autoencoder import QuantumAutoencoder
from .autograd import QAEModule
from torch.utils.data import Dataset, Sampler


class ClassicalAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, tf_init: bool = False):
        super(ClassicalAutoEncoder, self).__init__()

        self.fc1: torch.nn.Linear = torch.nn.Linear(input_dim, embedding_dim)
        self.fc2: torch.nn.Linear = torch.nn.Linear(embedding_dim, input_dim)

        if tf_init:
            # use the glorot uniform initialization that TensorFlow uses by default
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.zeros_(self.fc1.bias)

            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

    def get_embeddings(
        self, input_features: torch.Tensor, with_grad: bool = False
    ) -> torch.Tensor:
        with torch.set_grad_enabled(with_grad):
            tmp = torch.sigmoid(self.fc1(input_features))

            return tmp

    def load_latest_from_folder(self, folder_path: str):
        files = sorted(os.listdir(folder_path))
        newest_file = files[-1]

        checkpoint = torch.load(os.path.join(folder_path, newest_file))
        self.load_state_dict(checkpoint["model_state_dict"])


class RandomPartSampler(Sampler):
    def __init__(self, data_source: Dataset, example_cnt: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.example_cnt = example_cnt

    def __iter__(self):
        index_list = list(range(0, len(self.data_source)))
        random.shuffle(index_list)
        index_iter = iter(index_list[0 : self.example_cnt])

        return index_iter

    def __len__(self):
        return self.example_cnt


class HybridAutoencoder:
    def __init__(
        self,
        outer_cae: ClassicalAutoEncoder,
        inner_cae: Optional[ClassicalAutoEncoder],
        qae: QuantumAutoencoder,
    ):
        self.outer_cae: ClassicalAutoEncoder = outer_cae
        self.inner_cae: Optional[ClassicalAutoEncoder] = inner_cae
        self.qae = qae
        self.qae_module = QAEModule(qae)

    def calc_quantum_embeddings(self, input_values: torch.Tensor):
        with torch.no_grad():
            embeddings = self.qae.calc_embedding(
                self.outer_cae.get_embeddings(input_values) * math.pi,
                self.qae_module.encoder_params.reshape(1, -1).repeat(
                    input_values.shape[0], 1
                ),
                1024,
            )

        return embeddings

    def calc_fidelity(self, input1: np.ndarray, input2: np.ndarray) -> float:
        with torch.no_grad():
            input1_reduced: torch.Tensor = (
                self.outer_cae.get_embeddings(
                    torch.tensor(input1.reshape((1, -1)), dtype=torch.float32)
                )
                * math.pi
            )
            input2_reduced: torch.Tensor = (
                self.outer_cae.get_embeddings(
                    torch.tensor(input2.reshape((1, -1)), dtype=torch.float32)
                )
                * math.pi
            )

            fidelity = self.qae.calc_fidelity(
                input1_reduced.numpy().reshape((-1,)),
                input2_reduced.numpy().reshape((-1,)),
                self.qae_module.encoder_params,
                1024,
            )

            return fidelity

    def calc_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        distance_matrix = np.zeros((data.shape[0], data.shape[0]))

        for i in range(data.shape[0]):
            for j in range(0, i + 1):
                fidelity = self.calc_fidelity(data[i], data[j])
                distance = math.acos(math.sqrt(fidelity)) / (
                    math.pi / 2
                )  # Fubini-Study Metric + normalization

                # because the distance matrix is symmetric, only half of the values need to be computed
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix

    def calc_hybrid_reconstruction(self, input_values: torch.Tensor):
        with torch.no_grad():
            reconstructions = torch.sigmoid(
                self.outer_cae.fc2(
                    self.qae.calc_reconstructions(
                        self.outer_cae.get_embeddings(input_values) * math.pi,
                        self.qae_module.encoder_params.reshape(1, -1).repeat(
                            input_values.shape[0], 1
                        ),
                        1024,
                    )
                )
            )

        return reconstructions

    def calc_classical_reconstruction(self, input_values: torch.Tensor):
        if self.inner_cae is not None:
            with torch.no_grad():
                reconstructions = torch.sigmoid(
                    self.outer_cae.fc2(
                        self.inner_cae(
                            self.outer_cae.get_embeddings(input_values) * math.pi
                        )
                    )
                )
        else:
            raise ValueError()

        return reconstructions
