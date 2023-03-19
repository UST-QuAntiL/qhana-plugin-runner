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

from celery.utils.log import get_task_logger
from qiskit import IBMQ, Aer
from qiskit.utils import QuantumInstance


TASK_LOGGER = get_task_logger(__name__)


class QiskitBackends(enum.Enum):
    custom_ibmq = "custom_ibmq"
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

    def get_qiskit_backend(
        self,
        ibmq_token: str,
        custom_backend_name: str,
        shots: int,
    ) -> QuantumInstance:
        if self.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = self.name[4:]

            backend = Aer.get_backend(aer_backend_name)
        elif self.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            backend = provider.get_backend(self.name)
        elif self.name.startswith("custom_ibmq"):
            # Use custom IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            backend = provider.get_backend(custom_backend_name)
        else:
            raise NotImplementedError("Unknown qiskit backend specified!")

        return QuantumInstance(backend, shots=shots)
