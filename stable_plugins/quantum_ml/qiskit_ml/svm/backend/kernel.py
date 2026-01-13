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

from enum import Enum
from typing import Callable, List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit.primitives import BackendSamplerV2, StatevectorSampler
from qiskit.providers import BackendV2

try:
    from ...compat import ensure_qiskit_machine_learning_compat
except ImportError:
    from compat import ensure_qiskit_machine_learning_compat

ensure_qiskit_machine_learning_compat()

from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute


def _build_sampler(backend: BackendV2 | None, shots: int):
    if isinstance(backend, BackendV2):
        options = {"default_shots": shots} if shots else None
        return BackendSamplerV2(backend=backend, options=options)
    return StatevectorSampler(default_shots=shots or 1024)


def _decompose_feature_map(circuit: QuantumCircuit) -> QuantumCircuit:
    # Decompose blueprint feature maps so backends don't see custom instructions.
    for _ in range(3):
        if any(
            inst.operation.name.lower().endswith("featuremap") for inst in circuit.data
        ):
            circuit = circuit.decompose()
        else:
            break
    return circuit


class EntanglementPatternEnum(Enum):
    full = "Full"
    linear = "Linear"
    circular = "Circular"

    def get_pattern(self) -> str:
        return self.name


class KernelEnum(Enum):
    rbf = "Radial Basis Function"
    linear = "Linear"
    poly = "Polynomial"
    sigmoid = "Sigmoid"
    precomputed = "Precomputed"
    z_kernel = "Z Kernel"
    zz_kernel = "ZZ Kernel"
    pauli_kernel = "Pauli Kernel"

    def is_classical(self) -> bool:
        if (
            self.name == "rbf"
            or self.name == "linear"
            or self.name == "poly"
            or self.name == "sigmoid"
            or self.name == "precomputed"
        ):
            return True
        else:
            return False

    def get_kernel(
        self, **kwargs
    ) -> str | Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        if self.is_classical():
            return self.name
        else:
            return self.get_quantum_kernel(**kwargs)

    def get_quantum_kernel(
        self,
        backend,
        n_qubits: int,
        paulis: List[str],
        reps: int,
        entanglement_pattern: str,
        data_map_func: Callable[[np.array | List[float]], float],
        shots: int,
    ) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        if self == KernelEnum.z_kernel:
            feature_map = ZFeatureMap(
                feature_dimension=n_qubits,
                reps=reps,
                data_map_func=data_map_func,
            )  # This FeatureMap has no entanglement

        elif self == KernelEnum.zz_kernel:
            feature_map = ZZFeatureMap(
                feature_dimension=n_qubits,
                entanglement=entanglement_pattern,
                reps=reps,
                data_map_func=data_map_func,
            )

        elif self == KernelEnum.pauli_kernel:
            feature_map = PauliFeatureMap(
                feature_dimension=n_qubits,
                paulis=paulis,
                entanglement=entanglement_pattern,
                reps=reps,
                data_map_func=data_map_func,
            )

        else:
            raise NotImplementedError("Unkown kernel!")

        feature_map = _decompose_feature_map(feature_map)
        sampler = _build_sampler(backend, shots)
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
        return kernel.evaluate
