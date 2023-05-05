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

from torch.utils.data.dataset import Dataset
from torch import eye, Tensor


def digits2position(vec_of_digits: Tensor, n_positions: int):
    """One-hot encoding of a batch of vectors."""
    return eye(n_positions)[vec_of_digits]


class OneHotDataset(Dataset):
    """Dataset that converts the labels into one-hot encoded labels"""
    def __init__(self, data: Tensor, labels: Tensor, n_classes: int):
        self.data = data
        self.labels = digits2position(labels, n_classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]
