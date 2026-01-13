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
import qiskit
from numpy import ndarray
from typing import Callable

from pennylane_qiskit_compat import (
    ensure_qiskit_ibm_provider_compat,
    pennylane_qiskit_version_override,
)


def get_controlled_one_qubit_unitary(U: ndarray) -> Callable[[int, int], None]:
    ensure_qiskit_ibm_provider_compat()
    try:
        from qiskit.circuit import Instruction
    except Exception:
        Instruction = None
    if Instruction is not None and not hasattr(Instruction, "condition"):
        Instruction.condition = None
    qc = qiskit.QuantumCircuit(1)
    qc.unitary(U, [0])
    c_qc = qc.control()
    qc_transpiled = qiskit.transpile(
        c_qc,
        basis_gates=["u", "cx"],
        optimization_level=3,
    )
    with pennylane_qiskit_version_override():
        converted = qml.from_qiskit(qc_transpiled)

    def circuit(c_wire, t_wire):
        converted(wires=(t_wire, c_wire))

    return circuit
