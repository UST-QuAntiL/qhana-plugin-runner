from enum import Enum
import numpy as np
from typing import Callable, List


class DataMapsEnum(Enum):
    """
    Enum listing all available data maps
    """

    havlicek_kernel = "Havlíček Kernel"
    suzuki_kernel8 = "Suzuki Kernel 8"
    suzuki_kernel9 = "Suzuki Kernel 9"
    suzuki_kernel10 = "Suzuki Kernel 10"
    suzuki_kernel11 = "Suzuki Kernel 11"
    suzuki_kernel12 = "Suzuki Kernel 12"

    def get_data_mapping(self) -> Callable[[np.array | List[float]], float]:
        if self == self.havlicek_kernel:
            return havlicek_data_mapping
        elif self == self.suzuki_kernel8:
            return suzuki8_data_mapping
        elif self == self.suzuki_kernel9:
            return suzuki9_data_mapping
        elif self == self.suzuki_kernel10:
            return suzuki10_data_mapping
        elif self == self.suzuki_kernel11:
            return suzuki11_data_mapping
        elif self == self.suzuki_kernel12:
            return suzuki12_data_mapping
        else:
            raise NotImplementedError(f"Unknown data map!")


def havlicek_data_mapping(x: np.array | List[float]) -> float:
    result = 1
    if len(x) == 1:
        return x[0]
    for x_i in x:
        result *= np.pi - x_i
    return result


def suzuki8_data_mapping(x: np.array | List[float]) -> float:
    result = np.pi
    if len(x) == 1:
        return x[0]
    for x_i in x:
        result *= x_i
    return result


def suzuki9_data_mapping(x: np.array | List[float]) -> float:
    result = np.pi / 2.0
    if len(x) == 1:
        return x[0]
    for x_i in x:
        result *= 1 - x_i
    return result


def suzuki10_data_mapping(x: np.array | List[float]) -> float:
    if len(x) == 1:
        return x[0]
    square_diff = []
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x):
            if i != j:
                square_diff.append((x_i - x_j) * (x_j - x_i))
    mean = np.array(square_diff).mean()
    return np.pi * np.exp(mean / 8.0)


def suzuki11_data_mapping(x: np.array | List[float]) -> float:
    if len(x) == 1:
        return x[0]
    x = 1.0 / np.cos(x)
    return np.pi / 3.0 * np.product(x)


def suzuki12_data_mapping(x: np.array | List[float]) -> float:
    if len(x) == 1:
        return x[0]
    x = np.cos(x)
    return np.pi * np.product(x)
