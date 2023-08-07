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
from typing import List


def adaptive_ccnot(
    c_qubits: List[int], a_qubits: List[int], unclean_qubits: List[int], t_qubit: int
):
    """
    Chooses the ccnot variant with the least operations and executes it.
    :param c_qubits: list of control qubits
    :param a_qubits: list of ancilla qubits
    :param unclean_qubits: list of unclean qubits
    :param t_qubit: the target qubit
    :return: None
    """
    unclean_qubits = [] if unclean_qubits is None else unclean_qubits
    if len(c_qubits) == 0:
        qml.PauliX((t_qubit,))
    elif len(c_qubits) == 1:
        qml.CNOT((c_qubits[0], t_qubit))
    elif len(c_qubits) == 2:
        qml.Toffoli(wires=c_qubits + [t_qubit])
    elif len(a_qubits) == len(c_qubits) - 2:
        clean_ccnot(c_qubits, a_qubits, t_qubit)
    elif len(a_qubits) + len(unclean_qubits) == len(c_qubits) - 2:
        unclean_ccnot(c_qubits, a_qubits + unclean_qubits, t_qubit)
    elif len(a_qubits) > 0:
        one_ancilla_ccnot(c_qubits, a_qubits[0], t_qubit)
    else:
        raise NotImplementedError(
            f"There is no ccnot implemented for {len(c_qubits)} control qubits, {a_qubits} ancilla qubits and {unclean_qubits} unclean qubits."
        )


def one_ancilla_ccnot(c_qubits: List[int], a_qubit: List[int], t_qubit: int):
    """
    This consists of 4 steps
    1. Partition c_qubits into two sets c1 and c2
    2. Compute unclean ccnot(c1, c2, a_qubit)
    3. Compute unclean ccnot(c2+a_qubit, c1, t)
    4. Uncompute unclean ccnot(c1, c2, a_qubit)
    Given n many control qubits, an unclean ccnot needs n-2 ancilla qubits.
    Therefore, |c1| >= |c2| - 1 and |c2| >= |c1| - 2.
    If |c_qubits| is even => |c1| = |c_qubits|/2 and |c2| = |c_qubits|/2
    If |c_qubits| is uneven => |c1| = lower(|c_qubits|/2) and |c2| = ceil(|c_qubits|/2)
    """
    num_c1 = int(len(c_qubits) / 2)
    c1 = c_qubits[:num_c1]
    c2 = c_qubits[num_c1:]
    unclean_ccnot(c1, c2, a_qubit)
    unclean_ccnot(c2 + [a_qubit], c1, t_qubit)
    unclean_ccnot(c1, c2, a_qubit)


def clean_ccnot(c_qubits: List[int], a_qubits: List[int], t_qubit: int):
    """
    This ccnot operation only works, if the ancilla register has only zero values, i.e. it is clean
    :param c_qubits:
    :param a_qubits:
    :param t_qubit:
    :return:
    """
    if len(c_qubits) == 0:
        qml.PauliX((t_qubit,))
    elif len(c_qubits) == 1:
        qml.CNOT((c_qubits[0], t_qubit))
    elif len(c_qubits) == 2:
        qml.Toffoli(wires=c_qubits + [t_qubit])
    else:
        qml.Toffoli(wires=c_qubits[:2] + [a_qubits[0]])
        for i in range(2, len(c_qubits) - 1):
            qml.Toffoli(wires=[c_qubits[i]] + a_qubits[i - 2 : i])
        qml.Toffoli(wires=[c_qubits[-1], a_qubits[len(c_qubits) - 3], t_qubit])
        for i in range(len(c_qubits) - 2, 1, -1):
            qml.Toffoli(wires=[c_qubits[i]] + a_qubits[i - 2 : i])
        qml.Toffoli(wires=c_qubits[:2] + [a_qubits[0]])


def unclean_ccnot(c_qubits: List[int], a_qubits: List[int], t_qubit: int):
    """
    This ccnot operation works, even if the ancilla register has non zero values, i.e. it is not clean
    :param c_qubits:
    :param a_qubits:
    :param t_qubit:
    :return:
    """
    if len(c_qubits) == 0:
        qml.PauliX((t_qubit,))
    elif len(c_qubits) == 1:
        qml.CNOT(wires=c_qubits + [t_qubit])
    elif len(c_qubits) == 2:
        qml.Toffoli(wires=c_qubits + [t_qubit])
    else:
        n = len(c_qubits)
        qml.Toffoli(wires=[c_qubits[-1], a_qubits[-1], t_qubit])
        for i in range(-2, -n + 1, -1):
            qml.Toffoli(wires=[c_qubits[i], a_qubits[i], a_qubits[i + 1]])
        qml.Toffoli(wires=[c_qubits[0], c_qubits[1], a_qubits[-n + 2]])
        for i in range(-n + 2, -1):
            qml.Toffoli(wires=[c_qubits[i], a_qubits[i], a_qubits[i + 1]])
        qml.Toffoli(wires=[c_qubits[-1], a_qubits[-1], t_qubit])

        for i in range(-2, -n + 1, -1):
            qml.Toffoli(wires=[c_qubits[i], a_qubits[i], a_qubits[i + 1]])
        qml.Toffoli(wires=[c_qubits[0], c_qubits[1], a_qubits[-n + 2]])
        for i in range(-n + 2, -1):
            qml.Toffoli(wires=[c_qubits[i], a_qubits[i], a_qubits[i + 1]])
