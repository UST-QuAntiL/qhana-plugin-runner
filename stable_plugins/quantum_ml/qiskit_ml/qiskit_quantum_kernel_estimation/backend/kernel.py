# Copyright 2022 QHAna plugin runner contributors.
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

import enum
from typing import List

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit.primitives import BackendSamplerV2, StatevectorSampler
from qiskit.providers import BackendV2

try:
    from ...compat import ensure_qiskit_machine_learning_compat
except ImportError:  # pragma: no cover - fallback for direct plugin imports
    from compat import ensure_qiskit_machine_learning_compat

ensure_qiskit_machine_learning_compat()

from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute


def _build_sampler(backend: BackendV2 | None, shots: int):
    if isinstance(backend, BackendV2):
        options = {"default_shots": shots} if shots else None
        return BackendSamplerV2(backend=backend, options=options)
    return StatevectorSampler(default_shots=shots or 1024)


class EntanglementPatternEnum(enum.Enum):
    full = "full"
    linear = "linear"
    circular = "circular"

    def get_pattern(self) -> str:
        return self.name


class KernelEnum(enum.Enum):
    z_feature_map = "Z Kernel"
    zz_feature_map = "ZZ Kernel"
    pauli_feature_map = "Pauli Kernel"

    def get_kernel(
        self,
        backend,
        n_qbits: int,
        paulis: List[str],
        reps: int,
        entanglement_pattern: str,
        shots: int,
    ) -> FidelityQuantumKernel:
        if self == KernelEnum.z_feature_map:
            feature_map = ZFeatureMap(
                feature_dimension=n_qbits, reps=reps
            )  # This FeatureMap has no entanglement

        elif self == KernelEnum.zz_feature_map:
            feature_map = ZZFeatureMap(
                feature_dimension=n_qbits, entanglement=entanglement_pattern, reps=reps
            )

        elif self == KernelEnum.pauli_feature_map:
            feature_map = PauliFeatureMap(
                feature_dimension=n_qbits,
                paulis=paulis,
                entanglement=entanglement_pattern,
                reps=reps,
            )

        else:
            raise ValueError("Unkown kernel!")

        sampler = _build_sampler(backend, shots)
        fidelity = ComputeUncompute(sampler=sampler)
        return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
