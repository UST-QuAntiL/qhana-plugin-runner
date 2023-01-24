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
from .ccnot import adaptive_ccnot
from typing import List


def cc_increment_register(
    c_wires: List[int],
    target_wires: List[int],
    ancilla_wires: List[int],
    indicator_wire: int,
    unclean_wires: List[int] = None,
    indicator_is_zero: bool = True,
):
    """
    Increments a target register by one, if all the control qubits c_wires are 1
    :param c_wires: control qubits
    :param target_wires: target register
    :param ancilla_wires: ancilla qubits
    :param indicator_wire: qubit that indicates, whether the circuit should continue or not
    :param unclean_wires: unclean qubits (their state might not be |0>). They are used for ccnots.
    :param indicator_is_zero: if True, then the indicator_wire is in state |0>, else |1>.
    """
    if indicator_is_zero:
        qml.PauliX((indicator_wire,))  # indicator wire must be 1
    for i in range(len(target_wires) - 1, 0, -1):
        adaptive_ccnot(
            c_wires + [indicator_wire], ancilla_wires, unclean_wires, target_wires[i]
        )  # Increment
        adaptive_ccnot(
            c_wires + target_wires[i:], ancilla_wires, unclean_wires, indicator_wire
        )  # If we had flip from 0->1, then end computation, by setting ancilla wire to 0. Else 1->0 continue computation
        qml.PauliX((target_wires[i],))  # Only negated value of the bit is used later on
    adaptive_ccnot(
        c_wires + [indicator_wire], ancilla_wires, unclean_wires, target_wires[0]
    )  # flip overflow bit, if necessary
    adaptive_ccnot(
        c_wires, ancilla_wires, unclean_wires, indicator_wire
    )  # Reset ancilla wire to one | part 1
    adaptive_ccnot(
        c_wires + target_wires[1:], ancilla_wires, unclean_wires, indicator_wire
    )  # Reset ancilla wire to one | part 2

    for i in range(1, len(target_wires)):
        qml.PauliX((target_wires[i],))  # Reset the negated bits

    if indicator_is_zero:
        qml.PauliX((indicator_wire,))  # reset indicator to input value


def add_registers(
    control_reg: List[int],
    target_reg: List[int],
    indicator_wire: int,
    unclean_wires: List[int] = None,
    indicator_is_zero: bool = True
):
    if indicator_is_zero:
        qml.PauliX((indicator_wire,))
    for i in range(len(control_reg) - 1, -1, -1):
        cc_increment_register(
            [control_reg[i]],
            target_reg[: i + 2],
            control_reg[:i] + control_reg[i + 1 :],
            indicator_wire,
            unclean_wires=unclean_wires,
            indicator_is_zero=False,
        )
    if indicator_is_zero:
        qml.PauliX((indicator_wire,))
