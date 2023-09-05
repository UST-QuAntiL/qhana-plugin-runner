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

from os import PathLike
from typing import BinaryIO, Union

import torch
from torch.optim import Optimizer
from typing.io import IO


def save_optim_state(optim: Optimizer, file: Union[str, PathLike, BinaryIO, IO[bytes]]):
    torch.save(optim.state_dict(), file)


def load_optim_state(optim: Optimizer, file: Union[str, PathLike, BinaryIO, IO[bytes]]):
    optim.load_state_dict(torch.load(file))
