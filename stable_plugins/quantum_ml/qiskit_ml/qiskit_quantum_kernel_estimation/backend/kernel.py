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
from typing import List, Optional

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit.primitives import BackendSamplerV2
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator

from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute


def _decompose_feature_map(circuit: QuantumCircuit) -> QuantumCircuit:
    # Decompose blueprint feature maps so backends don't see custom
    # instructions.
    for _ in range(3):
        if any(
            inst.operation.name.lower().endswith("featuremap") for inst in circuit.data
        ):
            circuit = circuit.decompose()
        else:
            break
    return circuit


# Number of shots used when the caller does not request a specific shot count.
DEFAULT_SHOTS = 1024


def _build_sampler(backend: Optional[BackendV2], shots: int) -> BackendSamplerV2:
    """Return a sampler primitive for ``backend`` or a local Aer simulator."""
    if backend is None:
        backend = AerSimulator()
    return BackendSamplerV2(
        backend=backend, options={"default_shots": shots if shots else DEFAULT_SHOTS}
    )


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

        feature_map = _decompose_feature_map(feature_map)
        sampler = _build_sampler(backend, shots)
        fidelity = ComputeUncompute(sampler=sampler)
        return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
