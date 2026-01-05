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

import os
from typing import Any, Optional

from celery.utils.log import get_task_logger
from qiskit.providers.exceptions import QiskitBackendNotFoundError

TASK_LOGGER = get_task_logger(__name__)


def _runtime_config() -> tuple[str, Optional[str]]:
    channel = os.environ.get("QISKIT_IBM_CHANNEL") or os.environ.get("IBMQ_CHANNEL")
    instance = os.environ.get("QISKIT_IBM_INSTANCE") or os.environ.get("IBMQ_INSTANCE")
    return channel or "ibm_quantum", instance


def get_runtime_service(ibmq_token: str) -> Optional[Any]:
    """Get the Qiskit Runtime service from the token. If unavailable or invalid, return None.

    Args:
        ibmq_token: The IBMQ token to use.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as err:
        TASK_LOGGER.warning("Qiskit Runtime unavailable: %s", err)
        return None

    channel, instance = _runtime_config()
    try:
        if instance:
            return QiskitRuntimeService(
                channel=channel, token=ibmq_token, instance=instance
            )
        return QiskitRuntimeService(channel=channel, token=ibmq_token)
    except Exception as err:
        TASK_LOGGER.info("Runtime login failed: %s", err)
        return None


def get_backends(ibmq_token: str) -> Optional[list[Any]]:
    """Get the list of available backends for the given IBMQ token. If no service is found (e.g. invalid token), return None."""
    service = get_runtime_service(ibmq_token)
    if service is None:
        return None
    try:
        return service.backends()
    except Exception as err:
        TASK_LOGGER.info("Could not list IBM backends: %s", err)
        return None


def get_backend_names(ibmq_token: str) -> Optional[list[str]]:
    """Get the list of available backend names for the given IBMQ token. If no service is found (e.g. invalid token), return None."""
    backends = get_backends(ibmq_token)
    if backends is None:
        return None
    names = []
    for backend in backends:
        name = getattr(backend, "name", None)
        if callable(name):
            name = name()
        if name:
            names.append(name)
    return names


def get_qiskit_backend(backend: str, ibmq_token: str) -> Optional[Any]:
    """Get the backend with the given name from the Qiskit Runtime service. If no service is found (e.g. invalid token), return None."""
    service = get_runtime_service(ibmq_token)
    if service is None:
        return None
    try:
        return service.backend(backend)
    except QiskitBackendNotFoundError:
        TASK_LOGGER.info(f"Unknown qiskit backend specified: {backend}")
    except Exception as err:
        TASK_LOGGER.info("Could not load backend %s: %s", backend, err)
