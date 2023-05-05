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

from typing import Callable, Tuple
import numpy as np
from celery.utils.log import get_task_logger


TASK_LOGGER = get_task_logger(__name__)


class MaxCutClustering:
    """
    Interface for Clustering Object
    """

    def __init__(
        self,
        max_cut_solver: Callable[[np.array], Tuple[np.array, float]],
        number_of_clusters: int = 1,
    ):
        self.__max_cut_solver = max_cut_solver
        self.__number_of_clusters = number_of_clusters

    def __split_Matrix(
        self, adjacency_matrix: np.array, label: np.array, category: int
    ) -> np.array:
        """
        This function returns the submatrix that contains only rows and columns with the label equal to the category.
        :param adjacency_matrix: a matrix
        :param label: numpy array containing the label for each element associated with a column/row
        :param category: integer. Category to split the matrix by
        """
        # split the adjacency matrix in one smaller matrix. These matrix contains only adjacency with the right label
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
                        NSM[s, t] = adjacency_matrix[i, j]
        return NSM

    def create_cluster(self, adjacency_matrix: np.array) -> np.array:
        if self.__number_of_clusters == 1:
            return self.__label_via_max_cut(adjacency_matrix)
        else:
            # rekursiv Algorithmus for more than two clusters
            label = np.ones(adjacency_matrix.shape[0])
            label.astype(np.int)
            label_all = np.zeros(adjacency_matrix.shape[0])
            label_all.astype(np.int)
            label = self.__rekursivAlgorithmus(
                self.__number_of_clusters, adjacency_matrix, label, label_all, 1
            )
            # print("Done")
            return label.astype(np.int)

    def __rekursivAlgorithmus(
        self,
        iteration: int,
        adjacency_matrix: np.array,
        label: np.array,
        label_all: np.array,
        category: int,
    ) -> np.array:
        """
        Returns a list of labels for each node, by splitting a given graph into two subgraphs via a maxcut. It continues
        this process for each subgraph, until the recursion depth is equal to the parameter iteration. Nodes
        within the same subgraph get assigned the same label.
        :param iteration: integer determining the recursion's depth.
        :param adjacency_matrix: numpy array representing the weighted adjacency matrix of a graph/subgraph.
        :param label: numpy array with labels associated with the nodes in the graph/subgraph.
        :param label_all: numpy array with all the labels associated with the original graph.
        :param category: integer determining the category to split by.a
        """
        # rekursiv Algorithmus for more than two clusters
        if iteration == 0:
            return label
        else:
            if len(label) == 1 or len(label) == 0:
                return label

            if not adjacency_matrix.any():
                return np.zeros(adjacency_matrix.shape[0], dtype=int)

            new_label = self.__label_via_max_cut(adjacency_matrix)

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
            ones = self.__split_Matrix(adjacency_matrix, new_label, 1)
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
            zeros = self.__split_Matrix(adjacency_matrix, new_label, 0)

            self.__rekursivAlgorithmus(iteration - 1, zeros, label, label_all, 1)
            return label_all

    def __label_via_max_cut(self, adjacency_matrix: np.array) -> np.array:
        """
        Executes a max cut and returns labels for each node, depending on the max cut.
        :param adjacency_matrix: numpy array containing the weighted adjacency matrix of a graph.
        """
        (cut, cutValue) = self.__max_cut_solver(adjacency_matrix)

        return cut.astype(int)
