from abc import ABCMeta, abstractmethod
import numpy as np


class Clustering(metaclass=ABCMeta):
    """
    Interface for Clustering Object
    """

    @abstractmethod
    def create_cluster(self, position_matrix: np.matrix, similarity_matrix: np.matrix) -> np.matrix:
        pass

    # def get_keep_cluster_mapping(self):
    #     return self.keep_cluster_mapping
    #
    # def set_keep_cluster_mapping(self, keep_cluster_mapping):
    #     self.keep_cluster_mapping = keep_cluster_mapping
    #     return
