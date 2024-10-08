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

from typing import List
import numpy as np


def bitlist_to_int(bitlist: List[int]) -> int:
    if bitlist is None:
        return None
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def int_to_bitlist(num, length: int):
    binary = bin(num)[2:]
    result = [int(el) for el in binary]
    if len(result) > length:
        raise ValueError(
            f"Binary representation of {num} needs at least {len(result)} bits, but only got {length}."
        )
    result = [0] * (length - len(result)) + result
    return result


def is_binary(data: np.ndarray) -> bool:
    return np.array_equal(data, data.astype(bool))


def check_binary(data: np.ndarray, error_msg: str):
    if not is_binary(data):
        raise ValueError(error_msg)


def ceil_log2(value: float) -> int:
    return int(np.ceil(np.log2(value)))
