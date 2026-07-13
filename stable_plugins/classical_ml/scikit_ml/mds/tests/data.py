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

import json


def _distances(values):
    return json.dumps(
        [
            {"source": source, "target": target, "distance": distance}
            for (source, target), distance in values.items()
        ]
    )


ENTITY_DISTANCES = _distances(
    {
        ("e1", "e2"): 1.0,
        ("e1", "e3"): 2.0,
        ("e2", "e3"): 1.0,
    }
)

ZERO_DISTANCES = _distances(
    {
        ("e1", "e2"): 0.0,
        ("e1", "e3"): 1.0,
        ("e2", "e3"): 1.0,
    }
)

TINY_DISTANCES = _distances(
    {
        ("e1", "e2"): 0.0,
        ("e1", "e3"): 5e-7,
        ("e2", "e3"): 5e-7,
    }
)

INCOMPLETE_DISTANCES = _distances(
    {
        ("e1", "e2"): 1.0,
        ("e1", "e3"): 2.0,
    }
)
