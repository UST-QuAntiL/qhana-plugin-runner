from enum import Enum
from .max_cut_clustering import Clustering
from .bm_max_cut_clustering import BmMaxCut
from .sdp_max_cut_clustering import SdpMaxCut
from .classic_naive_max_cut_clustering import ClassicNaiveMaxCut
from .vqe_max_cut_clustering import VQEMaxCut


class MaxCutClusteringEnum(Enum):
    sdp = "Semi Definite Programming"
    bm = "Burer-Monteiro"
    classic_naive = "Classic Naive"
    vqe = "Variational Quantum Eigensolver (VQE)"

    def get_max_cut(self, number_of_clusters, **kwargs) -> Clustering:
        if self == MaxCutClusteringEnum.bm:
            return BmMaxCut(number_of_clusters)
        elif self == MaxCutClusteringEnum.sdp:
            return SdpMaxCut(number_of_clusters)
        elif self == MaxCutClusteringEnum.classic_naive:
            return ClassicNaiveMaxCut(number_of_clusters)
        elif self == MaxCutClusteringEnum.vqe:
            return VQEMaxCut(number_of_clusters, **kwargs)
