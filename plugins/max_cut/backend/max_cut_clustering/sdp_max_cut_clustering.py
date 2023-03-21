import numpy as np
from . import Clustering
import networkx as nx
from ..max_cut_solver import SdpMaxCutSolver
from celery.utils.log import get_task_logger


TASK_LOGGER = get_task_logger(__name__)


class SdpMaxCut(Clustering):
    def __init__(self, number_of_clusters=1):
        super().__init__()
        self.__number_of_clusters = number_of_clusters
        return

    def create_cluster(
        self, position_matrix: np.matrix, similarity_matrix: np.matrix
    ) -> np.matrix:
        if self.__number_of_clusters == 1:
            return self.__sdpMaxCutAlgo(similarity_matrix)
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
            new_label = self.__sdpMaxCutAlgo(similarity_matrix)

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

    def __create_graph(self, similarity_matrix: np.matrix) -> nx.Graph:
        probSize = similarity_matrix.shape[0]
        graph = nx.Graph()
        print(f"similarity_matrix: {similarity_matrix}")

        for i in range(0, probSize):
            for j in range(0, probSize):
                if i != j:
                    print(f"i: {i}, j: {j}")
                    print(f"similarity_matrix[{i}]: {similarity_matrix[i]}")
                    print(f"similarity_matrix[{i}][{j}]: {similarity_matrix[i][j]}")
                    graph.add_edge(i, j, weight=similarity_matrix[i][j])

        return graph

    def __sdpMaxCutAlgo(self, similarity_matrix: np.matrix) -> np.matrix:
        label = np.zeros(similarity_matrix.shape[0])

        if similarity_matrix.any() == np.zeros((similarity_matrix.shape)).any():
            return label.astype(np.int)

        # Create sdp max cut solver
        graph = self.__create_graph(similarity_matrix)

        # Solve

        solver = SdpMaxCutSolver(graph)
        (cutValue, cutEdges) = solver.solve()

        # Remove the max cut edges
        for u, v in cutEdges:
            graph.remove_edge(u, v)

        # Plot the graphs
        # from matplotlib import pyplot as plt
        # pos = nx.spring_layout(graph)
        # nx.draw(graph, pos)
        # labels = nx.get_edge_attributes(graph, 'weight')
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels = labels)
        # plt.savefig("CuttedGraph.png", format="PNG")
        # plt.clf()

        # define element 0 (left side of first cut) is cluster 0
        element0 = cutEdges[0][0]
        label[element0] = 0

        for node in graph.nodes():
            # if node has path to element 0, then cluster 0
            # if not then cluster 1
            if nx.algorithms.shortest_paths.generic.has_path(graph, element0, node):
                label[node] = 0
            else:
                label[node] = 1

        # print results
        print("solution:", str(cutEdges))
        print("solution objective:", str(cutValue))

        return label.astype(np.int)
