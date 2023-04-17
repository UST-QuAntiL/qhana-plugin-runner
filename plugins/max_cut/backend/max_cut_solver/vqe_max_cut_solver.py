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
        # print(f"__vqeAlgorithmus: qp\n{qp}")
        # print(qp.prettyprint())

        qubitOp, offset = qp.to_ising()
        # print("Offset:", offset)
        # print("Ising Hamiltonian:")
        # print(str(qubitOp))

        # construct SamplingVQE
        # print(f"qubitOp.num_qubits: {qubitOp.num_qubits}")
        ry = TwoLocal(
            qubitOp.num_qubits,
            "ry",
            "cz",
            reps=self.reps,
            entanglement=self.entanglement,
        )
        vqe = SamplingVQE(sampler=self.backend, ansatz=ry, optimizer=self.optimizer)

        # run SamplingVQE
        # print(f"__vqeAlgorithmus: qubitOp\n{qubitOp}")
        result = vqe.compute_minimum_eigenvalue(qubitOp)

        # print results
        x = max_cut.sample_most_likely(result.eigenstate)
        x = np.array(list(x) + [0] * (qubitOp.num_qubits - len(x)), dtype=np.int)
        # print("energy:", result.eigenvalue.real)
        # print("time:", result.optimizer_time)
        # print("max-cut objective:", result.eigenvalue.real + offset)
        # print("solution:", x)
        # print("solution objective:", qp.objective.evaluate(x))

        return x, qp.objective.evaluate(x)
