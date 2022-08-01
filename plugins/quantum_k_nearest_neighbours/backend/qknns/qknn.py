import numpy as np
from abc import abstractmethod
from enum import Enum


class QkNNEnum(Enum):
    qiskit_qknn = "qiskit qknn"

    def get_qknn(self, train_data, labels, k, backend, shots):
        if self == QkNNEnum.qiskit_qknn:
            from .qiskit_qknn import QiskitQkNN
            return QiskitQkNN(train_data, labels, k, backend, shots)

    def get_preferred_backend(self):
        if self == QkNNEnum.qiskit_qknn:
            return "qiskit"
        else:
            return "pennylane"


class QkNN:

    def __init__(self, train_data, train_labels, k, backend, shots):
        self.train_data = np.array(train_data)
        self.train_labels = np.array(train_labels)
        self.k = k
        self.backend = backend
        self.shots = shots
        self.fit()

    @abstractmethod
    def fit(self):
        """
        Abstract method to fit the qknn, if necessary
        """

    @abstractmethod
    def label_points(self, X):
        """
        Abstract method to label points
        """
