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

from celery.utils.log import get_task_logger
from qiskit import IBMQ

from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.ibmq.exceptions import IBMQAccountError
from qiskit.providers.ibmq.accountprovider import AccountProvider
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend, IBMQSimulator

TASK_LOGGER = get_task_logger(__name__)


def get_provider(ibmq_token: str) -> AccountProvider:
    try:
        provider = IBMQ.enable_account(ibmq_token)
    except IBMQAccountError as e:
        # Try to get provider from existing accounts
        providers = (p for p in IBMQ.providers() if p.credentials.token == ibmq_token)
        provider = next(iter(providers), None)
        if not provider:
            TASK_LOGGER.error("No IBMQ provider found!")
            raise e
    return provider


def get_backends(ibmq_token: str) -> list[IBMQBackend | IBMQSimulator]:
    provider = get_provider(ibmq_token)
    return provider.backends()


def get_qiskit_backend(backend: str, ibmq_token: str) -> IBMQBackend | IBMQSimulator:
    provider = get_provider(ibmq_token)
    try:
        return provider.get_backend(backend)
    except QiskitBackendNotFoundError:
        TASK_LOGGER.error(f"Unknown qiskit backend specified: {backend}")