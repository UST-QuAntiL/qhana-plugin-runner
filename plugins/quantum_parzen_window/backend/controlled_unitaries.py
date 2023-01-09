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


def get_controlled_one_qubit_unitary(U):
    qc = qiskit.QuantumCircuit(1)
    qc.unitary(U, [0])
    c_qc = qc.control()
    sv_backend = qiskit.Aer.get_backend("statevector_simulator")
    qc_transpiled = qiskit.transpile(
        c_qc,
        backend=sv_backend,
        basis_gates=sv_backend.configuration().basis_gates,
        optimization_level=3,
    )
    converted = qml.from_qiskit(qc_transpiled)

    def circuit(c_wire, t_wire):
        converted(wires=(t_wire, c_wire))

    return circuit
