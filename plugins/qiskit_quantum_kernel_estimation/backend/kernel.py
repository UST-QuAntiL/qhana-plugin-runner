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

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
import enum
from typing import List


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
        self, backend, n_qbits: int, paulis: List[str], reps: int, entanglement_pattern: str
    ) -> QuantumKernel:
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
                feature_dimension=n_qbits, paulis=paulis, entanglement=entanglement_pattern, reps=reps
            )

        else:
            raise ValueError("Unkown kernel!")

        return QuantumKernel(feature_map=feature_map, quantum_instance=backend)
