from pennylane import QuantumFunctionError


def check_wires_uniqueness(obj_with_wires, wire_types):
    for idx1, wire_type1 in enumerate(wire_types):
        wires1 = getattr(obj_with_wires, wire_type1 + '_wires')
        for wire_type2 in wire_types[idx1 + 1:]:
            wires2 = getattr(obj_with_wires, wire_type2 + '_wires')
            if any(wire in wires1 for wire in wires2):
                raise QuantumFunctionError(
                    f"The {wire_type1} wires must be different from the {wire_type2} wires"
                )


def check_num_wires(obj_with_wires, wire_types, num_wires, error_msgs):
    for w_type, n_wires, e_msg in zip(wire_types, num_wires, error_msgs):
        wires = getattr(obj_with_wires, w_type + '_wires')
        if len(wires) < n_wires:
            error = f"The number of {w_type} wires has to be greater or equal to {e_msg} Expected {n_wires}, but got {len(wires)}."
            raise QuantumFunctionError(error)
