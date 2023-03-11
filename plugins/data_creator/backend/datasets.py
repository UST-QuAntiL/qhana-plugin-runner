from enum import Enum
import numpy as np
from typing import List, Tuple


class DataTypeEnum(Enum):
    two_spirals = "Two Spirals"

    def get_data(self, num_train_points: int, num_test_points: int, **kwargs) -> Tuple[List, List, List, List]:
        if self == DataTypeEnum.two_spirals:
            data, labels = twospirals(num_train_points + num_test_points, **kwargs)
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            return data[indices[:num_train_points]].tolist(), labels[indices[:num_train_points]].tolist(), \
                   data[indices[num_train_points:]].tolist(), labels[indices[num_train_points:]].tolist()


def twospirals(n_points: int, noise=0.7, turns=1.52, **kwargs) -> Tuple[np.array, np.array]:
    """Returns the two spirals dataset."""
    n = np.sqrt(np.random.rand(n_points, 1)) * turns * (2 * np.pi)
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points).astype(int), np.ones(n_points).astype(int))),
    )
