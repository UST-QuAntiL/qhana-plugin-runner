from pennylane import QuantumFunctionError


def check_wires_uniqueness(obj_with_wires, wire_types):
    for idx1 in range(len(wire_types)):
        wire_type1 = wire_types[idx1]
        wires1 = getattr(obj_with_wires, wire_type1 + '_wires')
        for idx2 in range(idx1 + 1, len(wire_types)):
            wire_type2 = wire_types[idx2]
            wires2 = getattr(obj_with_wires, wire_type2 + '_wires')
            if any(wire in wires1 for wire in wires2):
                raise QuantumFunctionError(
                    f"The {wire_type1} wires must be different from the {wire_type2} wires"
                )


def check_num_wires(obj_with_wires, wire_types, num_wires, error_msgs):
    for i in range(len(wire_types)):
        wire_type = wire_types[i]
        wires = getattr(obj_with_wires, wire_type + '_wires')
        if len(wires) < num_wires[i]:
            error = f"The number of {wire_type} wires has to be greater or equal to {error_msgs[i]}"
            raise QuantumFunctionError(error)
