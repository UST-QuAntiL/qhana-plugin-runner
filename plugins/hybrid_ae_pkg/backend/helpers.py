from os import PathLike
from typing import BinaryIO, Union

import torch
from torch.optim import Optimizer
from typing.io import IO


def save_optim_state(optim: Optimizer, file: Union[str, PathLike, BinaryIO, IO[bytes]]):
    torch.save(optim.state_dict(), file)


def load_optim_state(optim: Optimizer, file: Union[str, PathLike, BinaryIO, IO[bytes]]):
    optim.load_state_dict(torch.load(file))
