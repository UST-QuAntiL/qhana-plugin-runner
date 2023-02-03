"""
    various classification algorithms
    
    c_svm:
        classical support vector machines using SKlearn
    
    qsvc:
        quantum support vector classifier (based on quantum kernel) using qiskit
        # quantum support vector machine with pennylane

    vqc:
        variational quantum classifier using qiskit
        # variational quantum classifier with pennylane

    NN:
        classic sklearn neural network

    QNN:
        hybrid qnn (already implemented with pennylane)

"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.optimize import NesterovMomentumOptimizer

import numpy as np


def classical_SVM(data, labels):
    clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    clf.fit(data, labels)
    # print( clf.predict(test_data) )
    return clf


def quantumkernel_SVM(data, labels, n_qubits):
    dev_kernel = qml.device("default.qubit", wires=n_qubits)

    projector = np.zeros((2**n_qubits, 2**n_qubits))
    projector[0, 0] = 1

    # quantum kernel
    @qml.qnode(dev_kernel)
    def kernel(x1, x2):
        AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

    # define the kernel matrix (needed for scikit-learn SVM)
    def kernel_matrix(A, B):
        return np.array([[kernel(a, b) for b in B] for a in A])

    # sklearn SVM
    svm = SVC(kernel=kernel_matrix).fit(data, labels)
    # svm.predict(test_data)
    return svm


def variational_quantum_classifier(data, labels, n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=4)

    def layer(W):
        """
        one layer of the quantum circuit
        """
        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
        qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])

    def statepreparation(x):
        """
        encode data
        """
        qml.BasisState(
            x, wires=[0, 1, 2, 3]
        )  # TODO basisstate input must be [..] consisting of 0s and 1s only

    @qml.qnode(dev)
    def circuit(weights, x):
        """
        variational quantum circuit
        """

        statepreparation(x)

        for W in weights:
            layer(W)

        return qml.expval(qml.PauliZ(0))

    def variational_classifier(weights, bias, x):
        """
        add classical bias parameter to the variational quantum circuit
        """
        return circuit(weights, x) + bias

    def square_loss(labels, predicitons):
        """
        loss function (standard square loss)
        """
        loss = 0

        for l, p in zip(labels, predicitons):
            loss += (l - p) ** 2

        loss /= len(labels)
        return loss

    def accuracy(labels, predictions):
        """
        accuracy given the target labels and the models predictions
        """
        accuracy = 0

        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                accuracy += 1

        accuracy /= len(labels)
        return accuracy

    def cost(weights, bias, data, labels):
        predictions = [variational_classifier(weights, bias, x) for x in data]
        return square_loss(labels, predictions)

    # shift labels to {-1, 1}
    labels = (labels - 1) * 2 - np.ones(len(labels))

    # random initialization
    np.random.seed(0)
    weight_init = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    # optimizer
    opt = NesterovMomentumOptimizer(0.5)

    batch_size = 5

    weights = weight_init
    bias = bias_init

    for it in range(25):
        # update weights
        batch_index = np.random.randint(0, len(X), (batch_size,))
        batch_data = data[batch_index]
        batch_labels = labels[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, batch_data, batch_labels)

        # compute accuracy
        # predictions = [np.sign(variational_classifier(weights, bias, x)) for x in data]
        # acc = accuracy( labels, predictions)

    def predict(weights, bias, data):
        predicitons = [np.sign(variational_classifier(weights, bias, x)) for x in data]
        return predicitons

    return predict


def classicalNN(data, labels):
    classifier = MLPClassifier(alpha=1e-4, hidden_layer_sizes=(100,), solver="adam")
    classifier.fit(data, labels)
    return classifier


# hybrid qnn implemented in model.py
# TODO move here? or move all this together in a models file?

# TODO test if this file works in plugin
# TODO modify visualization so it works with all models (pass model.predict as model?)
# TODO unify training and test for different models?
