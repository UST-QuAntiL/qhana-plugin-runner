from enum import Enum
from typing import Callable, Tuple
import numpy as np
import networkx as nx

from maxcut import MaxCutSDP
from maxcut import MaxCutBM
from .classic_naive_max_cut_solver import ClassicNaiveMaxCutSolver
from .vqe_max_cut_solver import VQEMaxCutSolver

from qiskit.primitives.sampler import BaseSampler
from qiskit.algorithms.optimizers.optimizer import Optimizer


def create_graph(adjacency_matrix: np.array) -> nx.Graph:
    graph = nx.Graph()

    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[0]):
            if i != j:
                graph.add_edge(i, j, weight=adjacency_matrix[i][j])

    return graph


class MaxCutSolverEnum(Enum):
    sdp = "Semi Definite Programming"
    bm = "Burer-Monteiro"
    classic_naive = "Classic Naive"
    vqe = "Variational Quantum Eigensolver (VQE)"

    def get_solver(self, **kwargs) -> Callable[[np.array], Tuple[np.array, float]]:
        if self == MaxCutSolverEnum.bm:
            return bm_maxcut
        elif self == MaxCutSolverEnum.sdp:
            return sdp_maxcut
        elif self == MaxCutSolverEnum.classic_naive:
            return classic_naive_maxcut
        elif self == MaxCutSolverEnum.vqe:

            def simpler_vqe(adjacency_matrix) -> (np.array, float):
                return vqe_maxcut(adjacency_matrix, **kwargs)

            return simpler_vqe


def sdp_maxcut(adjacency_matrix: np.array) -> (np.array, float):
    graph = create_graph(adjacency_matrix)
    sdp = MaxCutSDP(graph)
    cut = (sdp.get_results("cut") + 1) // 2
    cut_value = sdp.get_results("value")
    return cut, cut_value


def bm_maxcut(adjacency_matrix: np.array) -> (np.array, float):
    graph = create_graph(adjacency_matrix)
    bm = MaxCutBM(graph)
    cut = (bm.get_results("cut") + 1) // 2
    cut_value = bm.get_results("value")
    return cut, cut_value


def classic_naive_maxcut(adjacency_matrix: np.array) -> (np.array, float):
    graph = create_graph(adjacency_matrix)
    classic_naive = ClassicNaiveMaxCutSolver(graph)
    return classic_naive.solve()


def vqe_maxcut(
    adjacency_matrix: np.array,
    backend: BaseSampler,
    optimizer: Optimizer,
    reps: int,
    entanglement: str,
    **kwargs,
) -> (np.array, float):
    vqe = VQEMaxCutSolver(
        adjacency_matrix, backend, optimizer, reps=reps, entanglement=entanglement
    )
    return vqe.solve()
