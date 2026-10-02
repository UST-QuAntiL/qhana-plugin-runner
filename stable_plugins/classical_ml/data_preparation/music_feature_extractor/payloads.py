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

"""Helpers for assembling extraction payloads."""

from typing import Any

from .feature_groups import empty_feature_group_payload

SCHEMA_VERSION = "music-features/v1"


def empty_payload(
    source_hash: str,
    fmt: str,
    source_name: str | None,
    warnings: list[str],
    *,
    part_count: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "format": fmt,
            "sourceName": source_name,
            "hash": source_hash,
            "partCount": part_count,
        },
        "per_part": [],
        "global": {"meter_changes": [], "tempo_changes": []},
        "computed": {
            "ambitus": {"lowest": None, "highest": None, "semitone_span": None},
            "pitch_class_distribution": {
                "counts": {str(i): 0 for i in range(12)},
                "normalized": {str(i): 0.0 for i in range(12)},
            },
            "feature_groups": empty_feature_group_payload(),
        },
        "feature_vector": None,
        "feature_vector_schema": None,
        "warnings": warnings,
    }
