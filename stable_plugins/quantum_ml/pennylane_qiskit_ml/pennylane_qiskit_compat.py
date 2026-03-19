# Copyright 2024 QHAna plugin runner contributors.
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

from __future__ import annotations

from contextlib import contextmanager


def ensure_qiskit_ibm_provider_compat() -> None:
    """Patch qiskit symbols needed by qiskit-ibm-provider/runtime on qiskit>=2."""
    try:
        import qiskit.providers as providers
    except Exception:
        return

    if not hasattr(providers, "ProviderV1"):

        class ProviderV1:
            pass

        providers.ProviderV1 = ProviderV1

    try:
        import qiskit.providers.provider as provider_module
    except Exception:
        provider_module = None

    if provider_module and not hasattr(provider_module, "ProviderV1"):
        provider_module.ProviderV1 = providers.ProviderV1

    try:
        import qiskit.providers.backend as backend_module
    except Exception:
        backend_module = None

    if backend_module and not hasattr(backend_module, "BackendV1"):
        backend_v1 = getattr(backend_module, "BackendV2", None)
        if backend_v1 is None:

            class BackendV1:
                pass

            backend_v1 = BackendV1
        backend_module.BackendV1 = backend_v1
        if not hasattr(providers, "BackendV1"):
            providers.BackendV1 = backend_v1

    try:
        import qiskit_ibm_provider  # noqa: F401
    except Exception:
        import sys
        import types

        def _register_module(name: str, attrs: dict) -> None:
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules[name] = module

        sys.modules.pop("qiskit_ibm_provider", None)

        class IBMProvider:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "IBMQ backends are unavailable because qiskit-ibm-provider is "
                    "incompatible with qiskit>=2."
                )

            @staticmethod
            def saved_accounts():
                return {}

            @staticmethod
            def save_account(*args, **kwargs):
                raise RuntimeError(
                    "IBMQ backends are unavailable because qiskit-ibm-provider is "
                    "incompatible with qiskit>=2."
                )

        class IBMAccountError(RuntimeError):
            pass

        class AccountsError(RuntimeError):
            pass

        class IBMJobError(RuntimeError):
            pass

        _register_module("qiskit_ibm_provider", {"IBMProvider": IBMProvider})
        _register_module(
            "qiskit_ibm_provider.exceptions", {"IBMAccountError": IBMAccountError}
        )
        _register_module("qiskit_ibm_provider.accounts", {})
        _register_module(
            "qiskit_ibm_provider.accounts.exceptions", {"AccountsError": AccountsError}
        )
        _register_module("qiskit_ibm_provider.job", {"IBMJobError": IBMJobError})

    try:
        import qiskit_ibm_runtime  # noqa: F401
    except Exception:
        import sys
        import types

        def _register_runtime_module(name: str, attrs: dict) -> None:
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules[name] = module

        sys.modules.pop("qiskit_ibm_runtime", None)

        class QiskitRuntimeService:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "IBMQ runtime backends are unavailable because "
                    "qiskit-ibm-runtime is incompatible with qiskit>=2."
                )

        class RunnerResult(RuntimeError):
            pass

        _register_runtime_module(
            "qiskit_ibm_runtime", {"QiskitRuntimeService": QiskitRuntimeService}
        )
        _register_runtime_module(
            "qiskit_ibm_runtime.constants", {"RunnerResult": RunnerResult}
        )


@contextmanager
def pennylane_qiskit_version_override():
    try:
        import qiskit
    except Exception:
        yield
        return

    original_version = getattr(qiskit, "__version__", None)
    if not original_version:
        yield
        return

    qiskit.__version__ = "0.45.3"
    try:
        yield
    finally:
        qiskit.__version__ = original_version
