import pennylane as qml
import numpy as np
from ..utils import int_to_bitlist
from ..ccnot import adaptive_ccnot
from ..check_wires import check_wires_uniqueness, check_num_wires


class BinaryTreeNode:
    def __init__(self, bit_str, value, neg_sign=None):
        self.bit_str = bit_str
        self.value = value
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.neg_sign = neg_sign


class TreeLoader:
    def __init__(self, data, idx_wires, data_wires, ancilla_wires, unclean_wires=None):
        # 3 Ancillas needed
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data

        self.binary_trees = self.build_binary_tree_list()
        self.prepare_tree_list_values()

        self.idx_wires = idx_wires
        self.data_wires = data_wires
        self.ancilla_wires = ancilla_wires
        self.unclean_wires = [] if unclean_wires is None else unclean_wires  # unclean wires are like ancilla wires, but they are not guaranteed to be 0

        wire_types = ["idx", "data", "ancilla", "unclean"]
        num_wires = [int(np.ceil(np.log2(self.data.shape[0]))), int(np.ceil(np.log2(data.shape[1]))), 3]
        error_msgs = ["ceil(log2(size of train_data)).", "ceil(log2(datas' dimensionality)).", "3."]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-1], num_wires, error_msgs)

    def prepare_tree_values(self, node: BinaryTreeNode, sqrt_parent_value=1.):
        sqrt_value = np.sqrt(node.value)
        if node.parent is None:
            node.value = 0
        else:
            node.value = np.arccos(sqrt_value / sqrt_parent_value) * 2.

        if node.left_child is not None:
            self.prepare_tree_values(node.left_child, sqrt_value)
        if node.right_child is not None:
            self.prepare_tree_values(node.right_child, sqrt_value)

    def prepare_tree_list_values(self):
        for tree in self.binary_trees:
            self.prepare_tree_values(tree)

    def build_binary_tree_list(self):
        binary_trees = []
        if len(self.data.shape) == 1:
            binary_trees.append(self.build_binary_tree(self.data))
        else:
            for i in range(self.data.shape[0]):
                binary_trees.append(self.build_binary_tree(self.data[i]))
        return binary_trees

    def build_binary_tree(self, state):
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
                state[i]**2,
                neg_sign=(state[i] < 0.),
                )
            for i in range(len(state))]

        for depth in range(tree_depth):
            new_tree_nodes = []
            for i in range(0, len(tree_nodes), 2):
                new_tree_nodes.append(
                    BinaryTreeNode(tree_nodes[i].bit_str[:-1], tree_nodes[i].value + tree_nodes[i+1].value)
                )
                if tree_nodes[i].value != 0 or tree_nodes[i+1].value != 0:
                    # Set new node to the parent node
                    tree_nodes[i].parent = new_tree_nodes[-1]
                    tree_nodes[i+1].parent = new_tree_nodes[-1]
                    # Set new parent nodes children
                    new_tree_nodes[-1].left_child = tree_nodes[i]
                    new_tree_nodes[-1].right_child = tree_nodes[i+1]
            tree_nodes = new_tree_nodes
        return tree_nodes[-1]

    def get_sign_rotation(self, tree_node: BinaryTreeNode):
        if tree_node.left_child is not None:
            if tree_node.left_child.neg_sign is not None:
                """
                0: + +
                2pi: - -
                0 + z: + -
                2pi + z: - +
                """
                if tree_node.left_child.neg_sign and tree_node.right_child.neg_sign:
                    return 2.*np.pi, False
                elif tree_node.left_child.neg_sign and not tree_node.right_child.neg_sign:
                    return 2.*np.pi, True
                elif not tree_node.left_child.neg_sign and tree_node.right_child.neg_sign:
                    return 0, True
        return 0, False

    def qubit_rotations(self, qubit_idx: int, tree_node: BinaryTreeNode, right=True, tree_idx=0):
        if tree_node.left_child is not None:
            # rotate qubit
            sign_rot, use_z = self.get_sign_rotation(tree_node)
            if tree_node.parent is None:
                qml.CRY(tree_node.left_child.value + sign_rot, wires=(self.ancilla_wires[0], self.data_wires[qubit_idx]))
                if use_z:
                    qml.CZ(wires=(self.ancilla_wires[0], self.data_wires[qubit_idx]))
            else:
                adaptive_ccnot(
                    self.data_wires[:qubit_idx]+self.ancilla_wires[0:1],
                    [self.ancilla_wires[1]], self.unclean_wires+self.data_wires[qubit_idx+1:], self.ancilla_wires[2]
                )
                qml.CRY(tree_node.left_child.value+sign_rot, wires=(self.ancilla_wires[2], self.data_wires[qubit_idx]))
                if use_z:
                    qml.CZ((self.ancilla_wires[2], self.data_wires[qubit_idx]))
                adaptive_ccnot(
                    self.data_wires[:qubit_idx] + self.ancilla_wires[0:1],
                    [self.ancilla_wires[1]], self.unclean_wires + self.data_wires[qubit_idx + 1:], self.ancilla_wires[2]
                )


            # left child
            qml.PauliX((self.data_wires[qubit_idx],))
            # Reserve ancilla 1
            self.qubit_rotations(qubit_idx+1, tree_node.left_child, right=False)
            # Release ancilla 1
            qml.PauliX((self.data_wires[qubit_idx],))

            # right child
            # Reserve ancilla 1
            self.qubit_rotations(qubit_idx + 1, tree_node.right_child, right=True)
            # Release ancilla 1

    def load_tree(self, tree_idx: int):
        # print(f"log2_num_trees={log2_num_trees}")
        tree_idx_bits = int_to_bitlist(tree_idx, len(self.idx_wires))
        for i in range(len(self.idx_wires)):
            if tree_idx_bits[i] == 0:
                qml.PauliX((self.idx_wires[i],))

        # Reserved ancilla 0
        adaptive_ccnot(self.idx_wires, self.ancilla_wires[2:], self.data_wires+self.unclean_wires, self.ancilla_wires[0])
        self.qubit_rotations(0, self.binary_trees[tree_idx], tree_idx=tree_idx)
        # Release ancilla 0
        adaptive_ccnot(self.idx_wires, self.ancilla_wires[2:], self.data_wires+self.unclean_wires, self.ancilla_wires[0])

        for i in range(len(self.idx_wires)):
            if tree_idx_bits[i] == 0:
                qml.PauliX((self.idx_wires[i],))

    def circuit(self):
        log2_num_trees = int(np.log2(len(self.binary_trees)))
        log2_num_trees = max(1, log2_num_trees)
        for i in range(log2_num_trees, len(self.idx_wires)):
            qml.PauliX((self.idx_wires[i],))

        for tree_idx in range(len(self.binary_trees)):
            self.load_tree(tree_idx)

        for i in range(log2_num_trees, len(self.idx_wires)):
            qml.PauliX((self.idx_wires[i],))

    def get_circuit(self):
        return self.circuit

    def inv_circuit(self):
        self.get_inv_circuit()()

    def get_inv_circuit(self):
        return qml.adjoint(self.get_circuit())
