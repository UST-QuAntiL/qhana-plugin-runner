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

from typing import Any


def ensure_qiskit_machine_learning_compat() -> None:
    """Patch qiskit primitives for qiskit-machine-learning compatibility on qiskit>=2."""
    try:
        import qiskit.primitives as primitives
    except Exception:
        return

    def _ensure_attr(name: str, value: Any) -> None:
        if not hasattr(primitives, name):
            setattr(primitives, name, value)
        module = __import__("sys").modules.get("qiskit.primitives")
        if module and not hasattr(module, name):
            setattr(module, name, value)

    try:
        from qiskit.primitives import StatevectorSampler, BaseSamplerV1

        _ensure_attr("Sampler", StatevectorSampler)
        _ensure_attr("BaseSampler", BaseSamplerV1)
    except Exception:
        pass

    try:
        from qiskit.primitives import StatevectorEstimator, BaseEstimatorV1

        _ensure_attr("Estimator", StatevectorEstimator)
        _ensure_attr("BaseEstimator", BaseEstimatorV1)
    except Exception:
        pass

    try:
        from qiskit.primitives import utils as primitives_utils
    except Exception:
        return

    if not hasattr(primitives_utils, "_circuit_key"):

        def _circuit_key(circuit: Any) -> int:
            return id(circuit)

        primitives_utils._circuit_key = _circuit_key  # type: ignore[attr-defined]

    if not hasattr(primitives_utils, "init_observable"):

        def init_observable(observable: Any) -> Any:
            return observable

        primitives_utils.init_observable = init_observable  # type: ignore[attr-defined]
