# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pennylane as qml
import numpy as np
from typing import List, Tuple
from ..utils import int_to_bitlist, ceil_log2
from ..ccnot import adaptive_ccnot
from ..check_wires import check_wires_uniqueness, check_num_wires


"""
The TreeLoader implemented is from [0].
[0] I. Kerenidis, A. Prakash (2016), Quantum Recommendation Systems. arXiv. https://doi.org/10.48550/ARXIV.1603.08675
"""


class BinaryTreeNode:
    def __init__(
        self,
        bit_str: str,
        value: float,
        neg_sign: bool = None,
    ):
        self.bit_str = bit_str
        self.value = value
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.neg_sign = neg_sign


class TreeLoader:
    def __init__(
        self,
        data: np.ndarray,
        idx_wires: List[int],
        data_wires: List[int],
        ancilla_wires: List[int],
        unclean_wires: List[int] = None,
        control_wires: List[int] = None,
    ):
        # 3 Ancillas needed
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data

        self.idx_wires = [] if idx_wires is None else idx_wires
        self.data_wires = data_wires
        self.ancilla_wires = ancilla_wires
        self.unclean_wires = (
            [] if unclean_wires is None else unclean_wires
        )  # unclean wires are like ancilla wires, but they are not guaranteed to be 0
        self.control_wires = (
            [] if control_wires is None else control_wires
        )  # Only use TreeLoader, if all control_wires are |1>

        wire_types = ["idx", "data", "ancilla", "unclean", "control"]
        num_wires = [
            ceil_log2(self.data.shape[0]),
            ceil_log2(data.shape[1]),
            3,
        ]
        error_msgs = [
            "ceil(log2(size of train_data)).",
            "ceil(log2(datas' dimensionality)).",
            "3.",
        ]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-2], num_wires, error_msgs)

        self.binary_trees = self.build_binary_tree_list()
        self.prepare_tree_list_values()

    def prepare_tree_values(self, node: BinaryTreeNode, sqrt_parent_value: float = 1.0):
        """
        Changes values in binary tree, such that they can be directly used for a RY rotation.
        """
        sqrt_value = np.sqrt(node.value)
        if node.parent is None:
            node.value = 0
        else:
            node.value = np.arccos(sqrt_value / sqrt_parent_value) * 2.0

        if node.left_child is not None:
            self.prepare_tree_values(node.left_child, sqrt_value)
        if node.right_child is not None:
            self.prepare_tree_values(node.right_child, sqrt_value)

    def prepare_tree_list_values(self):
        for tree in self.binary_trees:
            self.prepare_tree_values(tree)

    def build_binary_tree_list(self) -> List[BinaryTreeNode]:
        binary_trees = []
        if len(self.data.shape) == 1:
            binary_trees.append(self.build_binary_tree(self.data))
        else:
            for data_entry in self.data:
                binary_trees.append(self.build_binary_tree(data_entry))
        return binary_trees

    def build_binary_tree(self, state: np.ndarray) -> BinaryTreeNode:
        """
        Create a binary tree. The leafs consist of the probabilities of the different states. Going up the tree the
        probabilities get added up. The root has a value equal to one.
        Example:
        Let |Psi> = 0.4 |00> + 0.4 |01> + 0.8 |10> + 0.2 |11> be the input state.
        Then the binary tree looks as follows
                   ┇1.00┇
                   ╱    ╲
                 ╱        ╲
            ┇0.32┇        ┇0.68┇
            ╱    ╲        ╱    ╲
        ┇0.16┇  ┇0.16┇┇0.64┇  ┇0.04┇

        If the left and right child have a value of zero, we remove them both.
        Example:
        Let |Psi> = 1.414 |00> + 1.414 |01> + 0. |10> + 0. |11> be the input state.
        Then the binary tree looks as follows
                   ┇1.00┇
                   ╱    ╲
                 ╱        ╲
            ┇1.00┇        ┇0.00┇
            ╱    ╲
        ┇0.50┇  ┇0.50┇

        :param state: The quantum state that needs to be loaded
        :return: returns a binary tree of the quantum state
        """
        tree_depth = int(np.log2(len(state)))
        tree_nodes = [
            BinaryTreeNode(
                "".join([str(el) for el in int_to_bitlist(i, tree_depth)]),
                state[i] ** 2,
                neg_sign=(state[i] < 0.0),
            )
            for i in range(len(state))
        ]

        for depth in range(tree_depth):
            new_tree_nodes = []
            for i in range(0, len(tree_nodes), 2):
                new_tree_nodes.append(
                    BinaryTreeNode(
                        tree_nodes[i].bit_str[:-1],
                        tree_nodes[i].value + tree_nodes[i + 1].value,
                    )
                )
                if tree_nodes[i].value != 0 or tree_nodes[i + 1].value != 0:
                    # Set new node to the parent node
                    tree_nodes[i].parent = new_tree_nodes[-1]
                    tree_nodes[i + 1].parent = new_tree_nodes[-1]
                    # Set new parent nodes children
                    new_tree_nodes[-1].left_child = tree_nodes[i]
                    new_tree_nodes[-1].right_child = tree_nodes[i + 1]
            tree_nodes = new_tree_nodes
        return tree_nodes[-1]

    def get_sign_rotation(self, tree_node: BinaryTreeNode) -> Tuple[float, bool]:
        """
        Depending, if an amplitude should be negative or positive, the RY rotations needs to be rotated by
        additional 2pi and a Z operations needs to be added. Since this concerns the amplitudes, only the leafs of
        the binary tree can have neg_sin == True.
        + +: 0
        - -: 2pi
        + -: 0 + z
        - +: 2pi + z
        """
        if tree_node.left_child is not None:
            if tree_node.left_child.neg_sign is not None:
                if tree_node.left_child.neg_sign and tree_node.right_child.neg_sign:
                    return 2.0 * np.pi, False
                elif tree_node.left_child.neg_sign and not tree_node.right_child.neg_sign:
                    return 2.0 * np.pi, True
                elif not tree_node.left_child.neg_sign and tree_node.right_child.neg_sign:
                    return 0, True
        return 0, False

    def qubit_rotations(self, qubit_idx: int, tree_node: BinaryTreeNode):
        """
        Recursively rotates the qubits in the data register to produce the correct state.
        """
        if tree_node.left_child is not None:
            # rotate qubit
            sign_rot, use_z = self.get_sign_rotation(tree_node)
            if tree_node.parent is None:
                qml.CRY(
                    tree_node.left_child.value + sign_rot,
                    wires=(self.ancilla_wires[0], self.data_wires[qubit_idx]),
                )
                if use_z:
                    qml.CZ(wires=(self.ancilla_wires[0], self.data_wires[qubit_idx]))
            else:
                adaptive_ccnot(
                    self.data_wires[:qubit_idx] + self.ancilla_wires[0:1],
                    [self.ancilla_wires[1]],
                    self.unclean_wires + self.data_wires[qubit_idx + 1 :],
                    self.ancilla_wires[2],
                )
                qml.CRY(
                    tree_node.left_child.value + sign_rot,
                    wires=(self.ancilla_wires[2], self.data_wires[qubit_idx]),
                )
                if use_z:
                    qml.CZ((self.ancilla_wires[2], self.data_wires[qubit_idx]))
                adaptive_ccnot(
                    self.data_wires[:qubit_idx] + self.ancilla_wires[0:1],
                    [self.ancilla_wires[1]],
                    self.unclean_wires + self.data_wires[qubit_idx + 1 :],
                    self.ancilla_wires[2],
                )

            # left child
            qml.PauliX((self.data_wires[qubit_idx],))
            self.qubit_rotations(qubit_idx + 1, tree_node.left_child)
            qml.PauliX((self.data_wires[qubit_idx],))

            # right child
            self.qubit_rotations(qubit_idx + 1, tree_node.right_child)

    def load_tree(self, tree_idx: int):
        """
        Loads the state described in the tree into the data register, given that the index register is equal to the
        tree's index.
        """
        if len(self.idx_wires) != 0:
            tree_idx_bits = int_to_bitlist(tree_idx, len(self.idx_wires))
            for tree_idx_bit_, idx_wire in zip(tree_idx_bits, self.idx_wires):
                if tree_idx_bit_ == 0:
                    qml.PauliX((idx_wire,))

        # Reserved ancilla 0
        adaptive_ccnot(
            self.control_wires + self.idx_wires,
            self.ancilla_wires[2:],
            self.data_wires + self.unclean_wires,
            self.ancilla_wires[0],
        )
        self.qubit_rotations(0, self.binary_trees[tree_idx])
        # Release ancilla 0
        adaptive_ccnot(
            self.control_wires + self.idx_wires,
            self.ancilla_wires[2:],
            self.data_wires + self.unclean_wires,
            self.ancilla_wires[0],
        )

        if len(self.idx_wires) != 0:
            for tree_idx_bit_, idx_wire in zip(tree_idx_bits, self.idx_wires):
                if tree_idx_bit_ == 0:
                    qml.PauliX((idx_wire,))

    def circuit(self):
        for tree_idx in range(len(self.binary_trees)):
            self.load_tree(tree_idx)

    def get_circuit(self):
        return self.circuit

    def inv_circuit(self):
        self.get_inv_circuit()()

    def get_inv_circuit(self):
        return qml.adjoint(self.get_circuit())

    @staticmethod
    def get_necessary_wires(data: np.ndarray) -> Tuple[float, float, float]:
        return (
            ceil_log2(data.shape[0]),
            ceil_log2(data.shape[1]),
            3,
        )
