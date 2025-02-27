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

from typing import Optional

from celery.utils.log import get_task_logger
from qiskit import IBMQ
from qiskit.providers import BackendV1
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit_ibm_provider import IBMBackend, IBMInputValueError, IBMProvider
from qiskit_ibm_provider.api.exceptions import RequestsApiError

TASK_LOGGER = get_task_logger(__name__)


def get_provider(ibmq_token: str) -> Optional[IBMProvider]:
    """Get the IBMQ provider from the token. If no provider is found (e.g. invalid token), return None.

    Args:
        ibmq_token: The IBMQ token to use.
    """
    try:
        provider = IBMProvider(ibmq_token)
    except RequestsApiError as err:
        if err.status_code == 401:
            TASK_LOGGER.info("User provided bad token!")
        else:
            TASK_LOGGER.info("Login failed!")
        return None
    except IBMInputValueError:
        TASK_LOGGER.info("Login failed!")
        return None

    return provider


def get_backends(ibmq_token: str) -> Optional[list[IBMBackend]]:
    """Get the list of available backends for the given IBMQ token. If no provider is found (e.g. invalid token), return None."""
    provider = get_provider(ibmq_token)
    if provider is None:
        return None
    return provider.backends()


def get_backend_names(ibmq_token: str) -> Optional[list[str]]:
    """Get the list of available backend names for the given IBMQ token. If no provider is found (e.g. invalid token), return None."""
    backends = get_backends(ibmq_token)
    if backends is None:
        return None
    return [backend.name for backend in backends]


def get_qiskit_backend(backend: str, ibmq_token: str) -> Optional[BackendV1]:
    """Get the backend with the given name from the IBMQ provider. If no provider is found (e.g. invalid token), return None."""
    provider = get_provider(ibmq_token)
    if provider is None:
        return None
    try:
        return provider.get_backend(backend)
    except QiskitBackendNotFoundError:
        TASK_LOGGER.info(f"Unknown qiskit backend specified: {backend}")
