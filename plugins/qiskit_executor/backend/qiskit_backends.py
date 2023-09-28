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

from celery.utils.log import get_task_logger
from qiskit import IBMQ

from qiskit.providers.ibmq.exceptions import IBMQAccountError


TASK_LOGGER = get_task_logger(__name__)


def get_qiskit_backend(backend: str, ibmq_token: str):
    try:
        provider = IBMQ.enable_account(ibmq_token)
    except IBMQAccountError as e:
        # Try to get provider from existing accounts
        providers = (p for p in IBMQ.providers() if p.credentials.token == ibmq_token)
        provider = next(iter(providers), None)
        if not provider:
            TASK_LOGGER.error("No IBMQ provider found!")
            raise e

    if backend.startswith("ibmq"):
        # Use IBMQ backend
        return provider.get_backend(backend)
    else:
        TASK_LOGGER.error("Unknown qiskit backend specified!")
