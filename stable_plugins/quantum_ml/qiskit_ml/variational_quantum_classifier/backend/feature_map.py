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

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
import enum
from typing import List


class EntanglementPatternEnum(enum.Enum):
    full = "full"
    linear = "linear"
    circular = "circular"

    def get_pattern(self) -> str:
        return self.name


class FeatureMapEnum(enum.Enum):
    z_feature_map = "Z Kernel"
    zz_feature_map = "ZZ Kernel"
    pauli_feature_map = "Pauli Kernel"

    def _decompose_feature_map(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # Decompose blueprint feature maps so backends don't see custom instructions.
        for _ in range(3):
            if any(
                inst.operation.name.lower().endswith("featuremap")
                for inst in circuit.data
            ):
                circuit = circuit.decompose()
            else:
                break
        return circuit

    def get_featuremap(
        self, n_qbits: int, paulis: List[str], reps: int, entanglement_pattern: str
    ) -> PauliFeatureMap:
        if self == FeatureMapEnum.z_feature_map:
            feature_map = ZFeatureMap(
                feature_dimension=n_qbits, reps=reps
            )  # This FeatureMap has no entanglement

        elif self == FeatureMapEnum.zz_feature_map:
            feature_map = ZZFeatureMap(
                feature_dimension=n_qbits, entanglement=entanglement_pattern, reps=reps
            )

        elif self == FeatureMapEnum.pauli_feature_map:
            feature_map = PauliFeatureMap(
                feature_dimension=n_qbits,
                paulis=paulis,
                entanglement=entanglement_pattern,
                reps=reps,
            )

        else:
            raise ValueError("Unkown feature map!")

        return self._decompose_feature_map(feature_map)
