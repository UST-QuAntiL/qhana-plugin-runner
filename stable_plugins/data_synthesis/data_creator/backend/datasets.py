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

from enum import Enum
from typing import List, Tuple

import numpy as np


class DataTypeEnum(Enum):
    two_spirals = "Two Spirals"
    checkerboard = "Checkerboard"

    def get_data(
        self, num_train_points: int, num_test_points: int, **kwargs
    ) -> Tuple[List, List, List, List]:
        """Returns specified dataset"""
        if self == DataTypeEnum.two_spirals:
            data, labels = twospirals(num_train_points + num_test_points, **kwargs)
        elif self == DataTypeEnum.checkerboard:
            data, labels = checkerboard(num_train_points + num_test_points, **kwargs)
        else:
            raise NotImplementedError
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        return (
            data[indices[:num_train_points]].tolist(),
            labels[indices[:num_train_points]].tolist(),
            data[indices[num_train_points:]].tolist(),
            labels[indices[num_train_points:]].tolist(),
        )


def twospirals(
    n_points: int, noise: float = 0.7, turns: float = 1.52, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the two spirals dataset."""
    n = np.sqrt(np.random.rand(n_points, 1)) * turns * (2 * np.pi)
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points).astype(int), np.ones(n_points).astype(int))),
    )


def checkerboard(n_points: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the checkerboard dataset."""
    rand_points = (np.random.rand(n_points, 2) * 2) - 1
    labels = np.zeros(n_points).astype(int)

    for i, (x, y) in enumerate(rand_points):
        # label by quadrant
        labels[i] = int(not ((x < 0 and y < 0) or (x >= 0 and y >= 0)))
        # push away from both axes
        x += 0.2 if x > 0 else -0.2
        y += 0.2 if y > 0 else -0.2
        rand_points[i] = x, y
    return rand_points, labels
