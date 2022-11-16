import pennylane as qml
from .ccnot import adaptive_ccnot


def cc_increment_register(c_wires, target_wires, toffolli_wires, ancilla_wire, unclean_wires=None, ancilla_is_zero=True):
    if ancilla_is_zero:
        qml.PauliX((ancilla_wire,))     # ancilla wire must be 1
    for i in range(len(target_wires)-1, 0, -1):
        adaptive_ccnot(c_wires+[ancilla_wire], toffolli_wires, unclean_wires, target_wires[i])  # Increment
        adaptive_ccnot(c_wires+target_wires[i:], toffolli_wires, unclean_wires, ancilla_wire)   # If we had flip from 0->1, then end computation, by setting ancilla wire to 0. Else 1->0 continue computation
        qml.PauliX((target_wires[i]))   # Only negated value of the bit is used later on
    adaptive_ccnot(c_wires+[ancilla_wire], toffolli_wires, unclean_wires, target_wires[0])  # flip overflow bit, if necessary
    adaptive_ccnot(c_wires, toffolli_wires, unclean_wires, ancilla_wire)                    # Reset ancilla wire to one | part 1
    adaptive_ccnot(c_wires+target_wires[1:], toffolli_wires, unclean_wires, ancilla_wire)   # Reset ancilla wire to one | part 2

    for i in range(1, len(target_wires)):
        qml.PauliX((target_wires[i],))  # Reset the negated bits

    if ancilla_is_zero:
        qml.PauliX((ancilla_wire,))     # reset ancilla to input value


def add_registers(control_reg, target_reg, ancilla_wires, unclean_wires=None, ancilla_is_zero=True):
    if ancilla_is_zero:
        qml.PauliX((ancilla_wires[-1]))
    for i in range(len(control_reg)-1, -1, -1):
        cc_increment_register(
            [control_reg[i]], target_reg[:i+2], control_reg[:i]+control_reg[i+1:]+ancilla_wires[:-1],
            ancilla_wires[-1], unclean_wires=unclean_wires, ancilla_is_zero=False
        )
    if ancilla_is_zero:
        qml.PauliX((ancilla_wires[-1]))
