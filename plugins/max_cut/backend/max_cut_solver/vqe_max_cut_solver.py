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

import numpy as np
from qiskit.primitives.sampler import BaseSampler
from qiskit.algorithms.optimizers.optimizer import Optimizer
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE


class VQEMaxCutSolver:
    def __init__(
        self,
        adjacency_matrix: np.array,
        backend: BaseSampler,
        optimizer: Optimizer,
        reps: int = 1,
        entanglement: str = "linear",
    ):
        self.adjacency_matrix = adjacency_matrix
        self.backend = backend
        self.optimizer = optimizer
        self.reps = reps
        self.entanglement = entanglement

    """
    Solves the max cut problem via VQE
    """

    def solve(self) -> (np.array, float):
        max_cut = Maxcut(self.adjacency_matrix)
        qp = max_cut.to_quadratic_program()

        qubitOp, offset = qp.to_ising()

        # construct SamplingVQE
        ry = TwoLocal(
            qubitOp.num_qubits,
            "ry",
            "cz",
            reps=self.reps,
            entanglement=self.entanglement,
        )
        vqe = SamplingVQE(sampler=self.backend, ansatz=ry, optimizer=self.optimizer)

        # run SamplingVQE
        result = vqe.compute_minimum_eigenvalue(qubitOp)

        x = max_cut.sample_most_likely(result.eigenstate)
        x = np.array(list(x) + [0] * (qubitOp.num_qubits - len(x)), dtype=np.int)

        return x, qp.objective.evaluate(x)
