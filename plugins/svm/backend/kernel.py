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
from typing import List, Callable, Optional
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel


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

        return QuantumKernel(feature_map=feature_map, quantum_instance=backend).evaluate
