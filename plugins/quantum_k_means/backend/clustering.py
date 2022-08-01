import enum
from abc import ABCMeta
from abc import abstractmethod
from typing import Tuple

import numpy as np
import pennylane as qml
from celery.utils.log import get_task_logger
from qiskit import IBMQ

from .quantumKMeans import (
    NegativeRotationQuantumKMeans,
    DestructiveInterferenceQuantumKMeans,
    StatePreparationQuantumKMeans,
    PositiveCorrelationQuantumKmeans,
)

TASK_LOGGER = get_task_logger(__name__)


class Clustering(metaclass=ABCMeta):
    """
    Interface for Clustering Object
    """

    @abstractmethod
    def create_cluster(self, position_matrix: np.ndarray) -> Tuple[np.ndarray, str]:
        pass


class QuantumBackends(enum.Enum):
    custom_ibmq = "custom_ibmq"
    aer_statevector_simulator = "aer_statevector_simulator"
    aer_qasm_simulator = "aer_qasm_simulator"
    ibmq_qasm_simulator = "ibmq_qasm_simulator"
    ibmq_santiago = "ibmq_santiago"
    ibmq_manila = "ibmq_manila"
    ibmq_bogota = "ibmq_bogota"
    ibmq_quito = "ibmq_quito"
    ibmq_belem = "ibmq_belem"
    ibmq_lima = "ibmq_lima"
    ibmq_armonk = "ibmq_armonk"

    @staticmethod
    def get_pennylane_backend(
        backend_enum: "QuantumBackends",
        ibmq_token: str,
        custom_backend_name: str,
        qubit_cnt: int,
    ) -> qml.Device:
        if backend_enum.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = backend_enum.name[4:]

            return qml.device("qiskit.aer", wires=qubit_cnt, backend=aer_backend_name)
        elif backend_enum.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=backend_enum.name,
                provider=provider,
            )
        elif backend_enum.name.startswith("custom_ibmq"):
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=custom_backend_name,
                provider=provider,
            )
        else:
            TASK_LOGGER.error("Unknown pennylane backend specified!")


class NegativeRotationQuantumKMeansClustering(Clustering):
    def __init__(
        self,
        number_of_clusters=2,
        max_qubits=2,
        shots_each=100,
        max_runs=10,
        relative_residual_amount=5,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        ibmq_custom_backend="",
    ):

        self.clusterAlgo = NegativeRotationQuantumKMeans()

        self.number_of_clusters = number_of_clusters
        self.max_qubits = max_qubits
        self.shots_each = shots_each
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def create_cluster(self, position_matrix: np.ndarray) -> Tuple[np.ndarray, str]:
        self.clusterAlgo.set_number_of_clusters(self.number_of_clusters)
        self.clusterAlgo.set_max_qubits(self.max_qubits)
        self.clusterAlgo.set_shots_each(self.shots_each)
        self.clusterAlgo.set_max_runs(self.max_runs)
        self.clusterAlgo.set_relative_residual_amount(self.relative_residual_amount)

        backend: qml.Device = QuantumBackends.get_pennylane_backend(
            self.backend, self.ibmq_token, self.ibmq_custom_backend, self.max_qubits
        )

        self.clusterAlgo.set_backend(backend)

        label = np.zeros(position_matrix.shape[0])

        # run
        clusterMapping, representative_circuit = self.clusterAlgo.Run(position_matrix)

        # write result into labels
        for i in range(0, len(label)):
            label[i] = int(clusterMapping[i])

        return label.astype(np.int), representative_circuit


class DestructiveInterferenceQuantumKMeansClustering(Clustering):
    def __init__(
        self,
        number_of_clusters=2,
        max_qubits=2,
        shots_each=100,
        max_runs=10,
        relative_residual_amount=5,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        ibmq_custom_backend="",
    ):

        self.clusterAlgo = DestructiveInterferenceQuantumKMeans()

        self.number_of_clusters = number_of_clusters
        self.max_qubits = max_qubits
        self.shots_each = shots_each
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def create_cluster(self, position_matrix: np.ndarray) -> Tuple[np.ndarray, str]:
        self.clusterAlgo.set_number_of_clusters(self.number_of_clusters)
        self.clusterAlgo.set_max_qubits(self.max_qubits)
        self.clusterAlgo.set_shots_each(self.shots_each)
        self.clusterAlgo.set_max_runs(self.max_runs)
        self.clusterAlgo.set_relative_residual_amount(self.relative_residual_amount)

        backend: qml.Device = QuantumBackends.get_pennylane_backend(
            self.backend, self.ibmq_token, self.ibmq_custom_backend, self.max_qubits
        )

        self.clusterAlgo.set_backend(backend)

        label = np.zeros(position_matrix.shape[0])

        # run
        clusterMapping, representative_circuit = self.clusterAlgo.Run(position_matrix)

        # write result into labels
        for i in range(0, len(label)):
            label[i] = int(clusterMapping[i])

        return label.astype(np.int), representative_circuit


class StatePreparationQuantumKMeansClustering(Clustering):
    def __init__(
        self,
        number_of_clusters=2,
        max_qubits=2,
        shots_each=100,
        max_runs=10,
        relative_residual_amount=5,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        ibmq_custom_backend="",
    ):

        self.clusterAlgo = StatePreparationQuantumKMeans()

        self.number_of_clusters = number_of_clusters
        self.max_qubits = max_qubits
        self.shots_each = shots_each
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def create_cluster(self, position_matrix: np.ndarray) -> Tuple[np.ndarray, str]:
        self.clusterAlgo.set_number_of_clusters(self.number_of_clusters)
        self.clusterAlgo.set_max_qubits(self.max_qubits)
        self.clusterAlgo.set_shots_each(self.shots_each)
        self.clusterAlgo.set_max_runs(self.max_runs)
        self.clusterAlgo.set_relative_residual_amount(self.relative_residual_amount)

        backend: qml.Device = QuantumBackends.get_pennylane_backend(
            self.backend, self.ibmq_token, self.ibmq_custom_backend, self.max_qubits
        )

        self.clusterAlgo.set_backend(backend)

        label = np.zeros(position_matrix.shape[0])

        # run
        clusterMapping, representative_circuit = self.clusterAlgo.Run(position_matrix)

        # write result into labels
        for i in range(0, len(label)):
            label[i] = int(clusterMapping[i])

        return label.astype(np.int), representative_circuit


class PositiveCorrelationQuantumKMeansClustering(Clustering):
    def __init__(
        self,
        number_of_clusters=2,
        shots_each=100,
        max_runs=10,
        relative_residual_amount=5,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token="",
        ibmq_custom_backend="",
    ):

        self.clusterAlgo = PositiveCorrelationQuantumKmeans()

        self.number_of_clusters = number_of_clusters
        self.shots_each = shots_each
        self.max_runs = max_runs
        self.relative_residual_amount = relative_residual_amount
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backend = ibmq_custom_backend
        return

    def create_cluster(self, position_matrix: np.ndarray) -> np.ndarray:
        backend = QuantumBackends.get_pennylane_backend(
            self.backend, self.ibmq_token, self.ibmq_custom_backend, 3
        )

        label = np.zeros(position_matrix.shape[0])

        # run
        clusterMapping = self.clusterAlgo.fit(
            position_matrix,
            self.number_of_clusters,
            self.max_runs,
            self.relative_residual_amount,
            backend,
            self.shots_each,
        )

        # write result into labels
        for i in range(0, len(label)):
            label[i] = int(clusterMapping[i])

        return label.astype(np.int)
