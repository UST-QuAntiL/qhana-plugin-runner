import enum
from abc import abstractmethod
from celery.utils.log import get_task_logger

from pennylane import Device
from typing import List
import numpy as np


TASK_LOGGER = get_task_logger(__name__)


class ClusteringEnum(enum.Enum):
    """
    Clustering Enum
    """

    negative_rotation = "Negative Rotation"
    destructive_interference = "Destructive Interference"
    state_preparation = "State Preparation"
    positive_correlation = "Positive Correlation"


    def get_cluster_algo(self,
                         backend: Device,
                         tol: float,
                         max_runs: int
                         ):
        if self == ClusteringEnum.negative_rotation:
            from .negative_rotation import NegativeRotationQuantumKMeans
            return NegativeRotationQuantumKMeans(backend, tol, max_runs)

        elif self == ClusteringEnum.destructive_interference:
            from .destructive_interference import DestructiveInterferenceQuantumKMeans
            return DestructiveInterferenceQuantumKMeans(backend, tol, max_runs)

        elif self == ClusteringEnum.positive_correlation:
            from .positive_correlation import PositiveCorrelationQuantumKmeans
            return PositiveCorrelationQuantumKmeans(backend, tol, max_runs)

        elif self == ClusteringEnum.state_preparation:
            from .state_preparation import StatePreparationQuantumKMeans
            return StatePreparationQuantumKMeans(backend, tol, max_runs)

        else:
            raise ValueError(f"Unkown clustering algorithm!")


class Clustering:
    def __init__(self, backend: Device, tol, max_runs):
        self.backend = backend
        self.max_qbits = backend.num_wires
        self.tol = tol
        self.max_runs = max_runs

    def Normalize(self, data):
        """
        Normalize the data, i.e. every entry of data has length 1.
        Note, that a copy of the data will be done.
        """
        return np.array([point / np.linalg.norm(point) for point in data])

    def Standardize(self, data):
        """
        Standardize all the points, i.e. they have zero mean and unit variance.
        Note that a copy of the data points will be created.
        """
        dataX = np.zeros(len(data))
        dataY = np.zeros(len(data))
        preprocessedData = np.array([None] * len(data))

        # create x and y coordinate arrays
        for i in range(0, len(data)):
            dataX[i] = data[i][0]
            dataY[i] = data[i][1]

        # make zero mean and unit variance, i.e. standardize
        tempDataX = (dataX - dataX.mean(axis=0)) / dataX.std(axis=0)
        tempDataY = (dataY - dataY.mean(axis=0)) / dataY.std(axis=0)

        # create tuples and normalize
        for i in range(0, len(data)):
            preprocessedData[i] = [tempDataX[i], tempDataY[i]]

        return preprocessedData

    def check_convergence(self, old_centroid_mapping, new_centroid_mapping):
        """
        Check if two centroid mappings are different. If they are similar enough, return True, otherwise return False.
        They are similar enough, if the number of different labels are less than
        the tolerance tol (a number between 0 and 1) times the total number of data points
        """
        n_different_labels = 0
        n_data_points = len(new_centroid_mapping)
        for i in range(0, len(old_centroid_mapping)):
            if old_centroid_mapping[i] != new_centroid_mapping[i]:
                n_different_labels += 1

        if n_different_labels - n_data_points * self.tol <= 0:
            return True
        else:
            return False

    @abstractmethod
    def init_centroids(self, k) -> np.ndarray:
        """
        Initialises centroids
        """

    @abstractmethod
    def prep_data(self, data) -> np.ndarray:
        """
        Prepare data
        """

    @abstractmethod
    def prep_data_for_circuit(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare data for circuit
        """

    @abstractmethod
    def prep_centroids(self, centroids) -> np.ndarray:
        """
        Prepare data
        """

    @abstractmethod
    def compute_new_centroid_mapping(self, preped_data, centroids) -> (List[int], int):
        """
        Returns new centroid mapping, depending on the prepared data and the current centroids
        """

    def get_mean_centroids_from_mapping(self,
                                        centroid_mapping: List[int],
                                        preped_data: np.ndarray,
                                        k: int
                                        ) -> np.ndarray:
        centroid_mapping = np.array(centroid_mapping)
        centroids = np.zeros((k, len(preped_data[0])))
        num_points = [0]*k
        # Sum points up
        for i in range(len(centroid_mapping)):
            c = centroid_mapping[i]
            centroids[c] += preped_data[i]
            num_points[c] += 1

        # Average sum
        for c in range(k):
            if num_points[c] == 0:
                centroids[c] = preped_data[c]
            else:
                centroids[c] = centroids[c] / num_points[c]

        return centroids

    def create_clusters(self, data: List[List[float]], k: int) -> np.ndarray:
        """
        Executes the quantum k means cluster algorithm on the given
        quantum backend and the specified circuit.
        The data needs to be 2D cartesian coordinates.
        We return a np.array with a mapping from data indizes to centeroid indizes,
        i.e. if we return a np.array [2, 0, 1, ...] this means:

        data vector with index 0 -> mapped to centroid with index 2
        data vector with index 1 -> mapped to centroid 0
        data vector with index 2 -> mapped to centroid 1
        """
        # The circuit version diverge on how the data and the centroids get preprocessed and how the centroid mappings
        # get computed, but the outer loop with the convergence criteria is the same.

        # Init centroids and Prepare data
        preped_centroids = self.prep_centroids(self.init_centroids(k))
        # Data gets prepared twice, since we need a version, to compute the new centroids after each iteration
        # and we might need a different version as input for the quantum circuits. Since the data points never change,
        # we compute both versions right here
        preped_data = self.prep_data(data)

        circuit_data = self.prep_data_for_circuit(preped_data)

        newCentroidMapping = np.zeros(len(data))
        iterations = 0
        not_converged = True
        globalAmountExecutedCircuits = 0
        while not_converged and iterations < self.max_runs:
            oldCentroidMapping = newCentroidMapping.copy()
            newCentroidMapping, amountExecutedCircuits = self.compute_new_centroid_mapping(circuit_data, preped_centroids)
            globalAmountExecutedCircuits += amountExecutedCircuits

            centroids = self.get_mean_centroids_from_mapping(newCentroidMapping, preped_data, k)
            preped_centroids = self.prep_centroids(centroids)

            not_converged = not self.check_convergence(oldCentroidMapping, newCentroidMapping)
            iterations += 1
            TASK_LOGGER.info(f"Iteration {iterations} done")

        return np.array(newCentroidMapping, dtype=int)


    # These methods were for debug purposes
    @abstractmethod
    def plot(self, preped_data, preped_centroids, centroid_mapping):
        """
        Plot data and centroids
        """

    @abstractmethod
    def simpler_plot(self, preped_data, preped_centroids, centroid_mapping):
        """
        Plot data and centroids
        """
