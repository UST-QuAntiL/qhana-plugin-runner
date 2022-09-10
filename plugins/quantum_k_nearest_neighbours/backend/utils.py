
def bitlist_to_int(bitlist):
    if bitlist is None:
        return None
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def int_to_bitlist(num, length: int):
    negative = False
    if num < 0:
        num *= -1
        negative = True
    binary = bin(num)[2:]
    result = [0]*length
    for i in range(-1, -len(binary)-1, -1):
        result[i] = int(binary[i])
    if negative:
        result[0] = 1
    return result


def check_if_values_are_binary(data):
    import numpy as np
    return np.array_equal(data, data.astype(bool))
