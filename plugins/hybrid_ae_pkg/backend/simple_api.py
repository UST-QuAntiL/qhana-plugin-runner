from typing import Tuple

import numpy as np
import pennylane as qml
import torch
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from .main import train_nn as training_loop
from .quantum.pl.pytorch.models.hybrid import HybridAutoencoder as PLHybridAutoencoder
from .quantum.pl.pytorch.training import training_loop as pl_training_loop
from .quantum.qiskit.autoencoder import QuantumAutoencoder as QKQuantumAutoencoder
from .quantum.qiskit.autograd import QAEModule
from .quantum.qiskit.hybrid import (
    ClassicalAutoEncoder as QKClassicalAutoEncoder,
    HybridAutoencoder as QKHybridAutoencoder,
)


def pennylane_hybrid_autoencoder(
    input_data: np.ndarray, q_num: int, embedding_size: int, qnn_name: str, steps: int, dev: qml.Device
) -> Tuple[np.ndarray, PLHybridAutoencoder, Optimizer, Optimizer]:
    model = PLHybridAutoencoder(input_data.shape[1], q_num, embedding_size, qnn_name, dev)
    c_optim = Adam(model.get_classical_parameters())
    q_optim = Adam(model.get_quantum_parameters(), lr=0.1)

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    pl_training_loop(
        model=model,
        shadow_model=None,
        train_input=input_tensor,
        train_target=input_tensor,
        train_label=None,
        test_input=None,
        test_target=None,
        test_label=None,
        grad_optis=[c_optim, q_optim],
        grad_free_opt=None,
        grad_free_opt_args=None,
        steps=steps,
        batch_size=1,
        checkpoint=None,
        save_checkpoints=False,
        db_id=None,
    )

    embeddings = model.embed(input_tensor)

    return embeddings.detach().numpy(), model, c_optim, q_optim


def qiskit_pytorch_autoencoder(
    input_data: np.ndarray,
    q_num: int,
    embedding_size: int,
    steps_c: int,
    steps_q: int,
    elements_per_iteration: int,
) -> np.ndarray:
    # train classical layers
    outer_cae = QKClassicalAutoEncoder(input_data.shape[1], q_num)
    optim = Adam(outer_cae.parameters())
    dataset = TensorDataset(torch.tensor(input_data, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    print("Training of the classical part:")
    print()
    training_loop(
        outer_cae, train_loader, None, steps_c, torch.nn.MSELoss(), optimizer=optim
    )

    # train quantum layers
    qae = QKQuantumAutoencoder(q_num, embedding_size, 1)
    qae_module = QAEModule(qae)
    optim = Adam(qae_module.parameters(), lr=0.1)

    embeddings = outer_cae.get_embeddings(dataset.tensors[0])
    embeddings_dataset = TensorDataset(embeddings)
    train_loader = DataLoader(
        embeddings_dataset,
        batch_size=1,
        sampler=RandomSampler(
            embeddings_dataset, num_samples=elements_per_iteration, replacement=True
        ),
    )

    def q_loss(output: torch.Tensor, _) -> torch.Tensor:
        fidelity = (2.0 * (output - 0.5)).clamp_min(0)
        qae_loss = (1 - fidelity).mean()

        return qae_loss

    print()
    print("Training of the quantum part:")
    print()
    training_loop(qae_module, train_loader, None, steps_q, q_loss, optimizer=optim)

    hybrid = QKHybridAutoencoder(outer_cae, None, qae)
    distance_matrix = hybrid.calc_distance_matrix(input_data)

    return distance_matrix


if __name__ == "__main__":
    pennylane_hybrid_autoencoder(np.zeros((1, 10)), 3, 2, "QNN3", 100)
    qiskit_pytorch_autoencoder(np.zeros((5, 10)), 3, 2, 1000, 10, 5)
