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

import pennylane as qml
from enum import Enum
from qiskit import IBMQ


class QuantumBackends(Enum):
    pennylane_default_qubit = "pennylane_default.qubit"
    aer_statevector_simulator = "aer_statevector_simulator"
    aer_qasm_simulator = "aer_qasm_simulator"
    ibmq_qasm_simulator = "ibmq_qasm_simulator"
    ibmq_santiago = "ibmq_santiago"
    ibmq_manila = "ibmq_manila"
    ibmq_bogota = "ibmq_bogota"
    ibmq_quito = "ibmq_quito"
    ibmq_belem = "ibmq_belem"
    ibmq_lima = "ibmq_lima"
    ibmq_armonk = "ibmq_armonk"
    custom_ibmq = "custom_ibmq"

    def get_pennylane_backend(
        self,
        ibmq_token: str,
        custom_backend_name: str,
        qubit_cnt: int,
        shots: int,
    ) -> qml.Device:
        if self.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = self.name[4:]
            return qml.device(
                "qiskit.aer", wires=qubit_cnt, backend=aer_backend_name, shots=shots
            )
        elif self.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=self.name,
                provider=provider,
                shots=shots,
            )
        elif self.name.startswith("custom_ibmq"):
            # Use custom IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=custom_backend_name,
                provider=provider,
                shots=shots,
            )
        elif self.name.startswith("pennylane"):
            return qml.device(self.value[10:], wires=qubit_cnt, shots=shots)
        else:
            # TASK_LOGGER.error
            raise NotImplementedError("Unknown pennylane backend specified!")
