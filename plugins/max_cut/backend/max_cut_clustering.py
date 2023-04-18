import numpy as np
from celery.utils.log import get_task_logger


TASK_LOGGER = get_task_logger(__name__)


class MaxCutClustering:
    """
    Interface for Clustering Object
    """

    def __init__(self, max_cut_solver, number_of_clusters=1):
        self.__max_cut_solver = max_cut_solver
        self.__number_of_clusters = number_of_clusters

    def __split_Matrix(
        self, similarity_matrix: np.array, label: np.array, category: int
    ) -> np.array:
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

    def create_cluster(self, similarity_matrix: np.array) -> np.array:
        if self.__number_of_clusters == 1:
            return self.__label_via_max_cut(similarity_matrix)
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
        similarity_matrix: np.array,
        label: np.array,
        label_all: np.array,
        category: int,
    ) -> np.array:
        # rekursiv Algorithmus for more than two clusters
        if iteration == 0:
            return label
        else:
            if len(label) == 1 or len(label) == 0:
                return label

            if not similarity_matrix.any():
                return np.zeros(similarity_matrix.shape[0], dtype=int)

            new_label = self.__label_via_max_cut(similarity_matrix)

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

    def __label_via_max_cut(self, adjacency_matrix: np.array) -> np.array:
        # Solve
        (cut, cutValue) = self.__max_cut_solver(adjacency_matrix)

        return cut.astype(int)
