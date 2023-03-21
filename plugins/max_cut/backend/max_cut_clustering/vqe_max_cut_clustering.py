import numpy as np
from . import Clustering
from qiskit.primitives.sampler import BaseSampler
from qiskit.algorithms.optimizers.optimizer import Optimizer
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.algorithms.optimizers import SPSA

from celery.utils.log import get_task_logger


TASK_LOGGER = get_task_logger(__name__)


class VQEMaxCut(Clustering):
    def __init__(
        self,
        number_of_clusters,
        backend: BaseSampler,
        optimizer: Optimizer,
        reps: int = 1,
        entanglement: str = "linear",
    ):
        super().__init__()
        self.__number_of_clusters = number_of_clusters
        self.__optimizer = optimizer
        self.__reps = reps
        self.__entanglement = entanglement
        self.__backend = backend

    def create_cluster(
        self, position_matrix: np.matrix, similarity_matrix: np.matrix
    ) -> np.matrix:
        if self.__number_of_clusters == 1:
            return self.__vqeAlgorithmus(similarity_matrix)
        else:
            # rekursiv Algorithmus for more than two clusters
            label = np.ones(similarity_matrix.shape[0])
            label.astype(np.int)
            label_all = np.zeros(similarity_matrix.shape[0])
            label_all.astype(np.int)
            label = self.__rekursivAlgorithmus(
                self.__number_of_clusters, similarity_matrix, label, label_all, 1
            )
            # print("Done")
            return label.astype(np.int)

    def __rekursivAlgorithmus(
        self,
        iteration: int,
        similarity_matrix: np.matrix,
        label: np.matrix,
        label_all: np.matrix,
        category: int,
    ) -> np.matrix:
        # rekursiv Algorithmus for more than two clusters
        if iteration == 0:
            return label
        else:
            if len(label) == 1 or len(label) == 0:
                return label
            new_label = self.__vqeAlgorithmus(similarity_matrix)

            z = -1
            check_label = np.ones(len(label))
            check_label.astype(np.int)
            for i in range(len(label)):
                check_label[i] = label[i]
            for i in range(len(label)):
                if check_label[i] == category:
                    z = z + 1
                    label_all[i] = label_all[i] + new_label[z] * pow(2, iteration - 1)
                    label[i] = new_label[z]
            TASK_LOGGER.info(
                "label after " + str(iteration) + " iteration :" + str(label_all)
            )

            # ones: rekursion only with ones labels in new label
            ones = self.__split_Matrix(similarity_matrix, new_label, 1)
            self.__rekursivAlgorithmus(iteration - 1, ones, label, label_all, 1)

            # change label for the zero cluster
            z = -1
            for i in range(len(label)):
                if check_label[i] == 1:
                    z = z + 1
                    if new_label[z] == 0:
                        label[i] = 1
                    else:
                        label[i] = 0
                else:
                    label[i] = 0

            # zeros: rekursion only with zero labels in new label
            zeros = self.__split_Matrix(similarity_matrix, new_label, 0)

            self.__rekursivAlgorithmus(iteration - 1, zeros, label, label_all, 1)
            return label_all

    def __split_Matrix(
        self, similarity_matrix: np.matrix, label: np.matrix, category: int
    ) -> np.matrix:
        # split the similarity matrix in one smaller matrix. These matrix contains only similarities with the right label
        npl = 0
        for i in range(len(label)):
            if label[i] == category:
                npl = npl + 1

        NSM = np.zeros((npl, npl))
        s = -1
        t = -1
        for i in range(len(label)):
            if label[i] == category:
                s += 1
                t = -1
                for j in range(len(label)):
                    if label[j] == category:
                        t += 1
                        NSM[s, t] = similarity_matrix[i, j]
        return NSM

    def __vqeAlgorithmus(self, similarity_matrix: np.matrix) -> np.matrix:
        max_cut = Maxcut(similarity_matrix)
        qp = max_cut.to_quadratic_program()
        print(qp.prettyprint())

        qubitOp, offset = qp.to_ising()
        print("Offset:", offset)
        print("Ising Hamiltonian:")
        print(str(qubitOp))

        # construct SamplingVQE
        print(f"qubitOp.num_qubits: {qubitOp.num_qubits}")
        ry = TwoLocal(
            qubitOp.num_qubits,
            "ry",
            "cz",
            reps=self.__reps,
            entanglement=self.__entanglement,
        )
        vqe = SamplingVQE(sampler=self.__backend, ansatz=ry, optimizer=self.__optimizer)

        # run SamplingVQE
        result = vqe.compute_minimum_eigenvalue(qubitOp)

        # print results
        x = max_cut.sample_most_likely(result.eigenstate)
        x = np.array(list(x) + [0] * (qubitOp.num_qubits - len(x)), dtype=np.int)
        print("energy:", result.eigenvalue.real)
        print("time:", result.optimizer_time)
        print("max-cut objective:", result.eigenvalue.real + offset)
        print("solution:", x)
        print("solution objective:", qp.objective.evaluate(x))

        return x

        # if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
        #     label = np.zeros(similarity_matrix.shape[0])
        #     return label.astype(np.int)
        # qubitOp, offset = max_cut.get_operator(similarity_matrix)
        # seed = 10598
        #
        # spsa = SPSA(max_trials=self.__max_trials)
        # ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=self.__reps, entanglement=self.__entanglement)
        # vqe = VQE(qubitOp, ry, spsa, quantum_instance=self.__backend)
        #
        # # run VQE
        # result = vqe.run(self.__backend)
        #
        # # print results
        # x = sample_most_likely(result.eigenstate)
        # print('energy:', result.eigenvalue.real)
        # print('time:', result.optimizer_time)
        # print('max-cut objective:', result.eigenvalue.real + offset)
        # print('solution:', max_cut.get_graph_solution(x))
        # print('solution objective:', max_cut.max_cut_value(x, similarity_matrix))
        # solution = max_cut.get_graph_solution(x)
        # return solution.astype(np.int)
