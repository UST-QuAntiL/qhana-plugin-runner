from itertools import islice
from typing import Iterator


def grouper(iterator: Iterator, n: int) -> Iterator[list]:
    while chunk := list(islice(iterator, n)):
        yield chunk
