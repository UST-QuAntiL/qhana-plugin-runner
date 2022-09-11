# Copyright 2021 QHAna plugin runner contributors.
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
from abc import abstractmethod
from typing import List
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import QuantumInstance
import numpy as np
import enum
# from celery.utils.log import get_task_logger
#
# TASK_LOGGER = get_task_logger(__name__)


class EntanglementPatternEnum(enum.Enum):
    full = "full"
    linear = "linear"
    circular = "circular"

    def get_pattern(self) -> str:
        return self.name


class KernelEnum(enum.Enum):
    z_feature_map = 'ZFeatureMap'
    zz_feature_map = 'ZZFeatureMap'
    pauli_feature_map = 'PauliFeatureMap'

    def get_kernel(self,
                   backend,
                   n_qbits: int,
                   reps: int,
                   entanglement_pattern: str) -> QuantumKernel:
        # backend = QuantumInstance(backend, shots)
        if self == KernelEnum.z_feature_map:
            featureMap = ZFeatureMap(feature_dimension=n_qbits, reps=reps)  # This FeatureMap has no entanglement

        elif self == KernelEnum.zz_feature_map:
            featureMap = ZZFeatureMap(feature_dimension=n_qbits, entanglement=entanglement_pattern, reps=reps)

        elif self == KernelEnum.pauli_feature_map:
            featureMap = PauliFeatureMap(feature_dimension=n_qbits, entanglement=entanglement_pattern, reps=reps)

        else:
            raise ValueError(f"Unkown kernel!")

        return QuantumKernel(feature_map=featureMap, quantum_instance=backend)

