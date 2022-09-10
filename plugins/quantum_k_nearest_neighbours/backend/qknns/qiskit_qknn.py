from .qknn import QkNN
from qiskit_quantum_knn.qknn import QKNeighborsClassifier
from qiskit_quantum_knn.encoding import analog
from qiskit import aqua
import numpy as np


class QiskitQkNN(QkNN):
    def __init__(self, train_data, train_labels, k, backend, shots):
        super(QiskitQkNN, self).__init__(train_data, train_labels, k, backend, shots)

        instance = aqua.QuantumInstance(backend, shots=self.shots)
        self.qknn = QKNeighborsClassifier(
            n_neighbors=self.k,
            quantum_instance=instance
        )
        n_variables = int(np.power(2, np.ceil(np.log2(self.train_data.shape[1]))))  # should be positive power of 2

        # encode data
        if self.train_data.shape[1] < n_variables:
            self.train_data = np.pad(self.train_data, [(0, 0), (0, n_variables - self.train_data.shape[1])],
                                     mode='constant',
                                     constant_values=0)

        encoded_train_data = analog.encode(self.train_data)

        self.train_data = encoded_train_data

    def label_points(self, X):
        X = np.pad(X, [(0, 0), (0, self.train_data.shape[1] - X.shape[1])], mode='constant', constant_values=0)
        encoded_X = analog.encode(X)
        labels = self.qknn.predict(encoded_X)
        return labels
