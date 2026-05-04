# Copyright 2026 QHAna plugin runner contributors.
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

"""Shared utility helpers for music feature extraction."""

from hashlib import sha256
from statistics import fmean, pstdev
from typing import Any, Sequence
from xml.etree import ElementTree as ET


def mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(fmean(values))


def std(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return float(pstdev(values))


def clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def clip_midi_velocity(value: float) -> int:
    rounded = int(round(value))
    if rounded < 0:
        return 0
    if rounded > 127:
        return 127
    return rounded


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def unique_changes(
    items: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=measure_sort_key)
    return unique


def measure_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    measure = item.get("measure")
    if isinstance(measure, int):
        return (measure, "")
    if isinstance(measure, str) and measure.isdigit():
        return (int(measure), "")
    return (10**9, str(measure))


def parse_measure_number(value: str | None) -> int | str | None:
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    return value


def get_text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    text = element.text.strip()
    return text if text else None


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def strip_tag(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()
