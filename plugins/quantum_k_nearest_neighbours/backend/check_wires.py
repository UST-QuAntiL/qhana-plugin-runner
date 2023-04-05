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

from pennylane import QuantumFunctionError
from typing import List


def check_wires_uniqueness(obj_with_wires: object, wire_types: List[str]):
    for idx1, wire_type1 in enumerate(wire_types):
        wires1 = getattr(obj_with_wires, wire_type1 + "_wires")
        for wire_type2 in wire_types[idx1 + 1 :]:
            wires2 = getattr(obj_with_wires, wire_type2 + "_wires")
            if any(wire in wires1 for wire in wires2):
                raise QuantumFunctionError(
                    f"The {wire_type1} wires must be different from the {wire_type2} wires"
                )


def check_num_wires(
    obj_with_wires: object,
    wire_types: List[str],
    num_wires: List[int],
    error_msgs: List[str],
):
    for w_type, n_wires, e_msg in zip(wire_types, num_wires, error_msgs):
        wires = getattr(obj_with_wires, w_type + "_wires")
        if len(wires) < n_wires:
            error = f"The number of {w_type} wires has to be greater or equal to {e_msg} Expected {n_wires}, but got {len(wires)}."
            raise QuantumFunctionError(error)
