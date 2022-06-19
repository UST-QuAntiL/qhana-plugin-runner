import pennylane as qml
from pennylane import numpy as np

# PyTorch
import torch
import torch.nn as nn

#######################################
# VARIATIONAL QUANTUM CIRCUIT

# define quantum layers


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT."""
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


# dressed quantum circuit
class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, n_qubits, quantum_device, q_depth):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.pre_net = nn.Linear(2, n_qubits)
        self.q_params = nn.Parameter(0.01 * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 2)

        # define circuit
        @qml.qnode(quantum_device, interface="torch")
        def quantum_net(q_input_features, q_weights_flat):
            """
            The variational quantum circuit.
            """

            # Reshape weights
            q_weights = q_weights_flat.reshape(q_depth, n_qubits)

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            H_layer(n_qubits)

            # Embed features in the quantum node
            RY_layer(q_input_features)

            # Sequence of trainable variational layers
            for k in range(q_depth):
                entangling_layer(n_qubits)
                RY_layer(q_weights[k])

            # Expectation values in the Z basis
            exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
            return tuple(exp_vals)

        self.q_net = quantum_net
        self.n_qubits = n_qubits

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        # q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = (
                self.q_net(elem, self.q_params).float().unsqueeze(0)
            )  # quantum_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)
