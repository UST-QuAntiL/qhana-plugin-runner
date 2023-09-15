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
from typing import Optional

from celery.utils.log import get_task_logger
from qiskit import IBMQ

from qiskit.providers.ibmq.exceptions import IBMQAccountError


TASK_LOGGER = get_task_logger(__name__)


def get_qiskit_backend_enum(ibmq_token: Optional[str] = None) -> enum.Enum:
    if not ibmq_token:
        return enum.Enum(
            "Backend",
            {"custom_ibmq": "custom_ibmq", "ibmq_qasm_simulator": "ibmq_qasm_simulator"},
            type=QiskitBackends,
        )
    try:
        provider = IBMQ.enable_account(ibmq_token)
    except IBMQAccountError as e:
        # Try to get provider from existing accounts
        providers = (p for p in IBMQ.providers() if p.credentials.token == ibmq_token)
        provider = next(iter(providers), None)
        if not provider:
            TASK_LOGGER.error("No IBMQ provider found!")
            raise e
    backends_dict = {"custom_ibmq": "custom_ibmq"}
    for backend in provider.backends():
        backends_dict[backend.name()] = backend.name()
    return enum.Enum(
        "Backend",
        backends_dict,
        type=QiskitBackends,
    )


class QiskitBackends:
    def get_qiskit_backend(
        self,
        ibmq_token: str,
        custom_backend_name: str,
    ):
        try:
            provider = IBMQ.enable_account(ibmq_token)
        except IBMQAccountError as e:
            # Try to get provider from existing accounts
            providers = (p for p in IBMQ.providers() if p.credentials.token == ibmq_token)
            provider = next(iter(providers), None)
            if not provider:
                TASK_LOGGER.error("No IBMQ provider found!")
                raise e

        if self.name.startswith("ibmq"):
            # Use IBMQ backend
            return provider.get_backend(self.name)
        elif self.name.startswith("custom_ibmq"):
            # Use custom IBMQ backend
            return provider.get_backend(custom_backend_name)
        else:
            TASK_LOGGER.error("Unknown qiskit backend specified!")
