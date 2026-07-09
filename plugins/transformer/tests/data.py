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

"""Test data for the similarity to distance transformers plugin.

Contains mock element similarities for a single-value attribute ("color") and
a multi-value attribute ("tags"). The element similarities for "color" contain a
null similarity for the (blue, blue) pair to exercise missing data handling.
"""

import math

TEST_DATA = {
    "element-similarities": {
        "color.json": r'[{"source":"red","target":"red","similarity":1.0},{"source":"red","target":"blue","similarity":0.0},{"source":"blue","target":"blue","similarity":null}]',
        "tags.json": r'[{"source":"x","target":"x","similarity":1.0},{"source":"x","target":"y","similarity":0.5},{"source":"y","target":"y","similarity":1.0}]',
    }
}


def _distances(values, transformer):
    """Calculates the expected distances from the simulated similarity data."""
    res = []
    for (source, target), sim in values.items():
        dist = None
        if sim is not None:
            if transformer == "linear_inverse":
                dist = 1.0 - sim
            elif transformer == "exponential_inverse":
                dist = math.exp(-sim)
            elif transformer == "gaussian_inverse":
                dist = math.exp(-sim * sim)
            elif transformer == "polynomial_inverse":
                # alpha=1.0, beta=1.0 as hardcoded in tasks.py
                dist = 1.0 / (1.0 + sim)
            elif transformer == "square_inverse":
                # max_sim = 1.0
                dist = (1.0 / math.sqrt(2.0)) * math.sqrt(2.0 - 2 * sim)

        res.append({"source": source, "target": target, "distance": dist})
    return res


_COLOR_SIMS = {
    ("red", "red"): 1.0,
    ("red", "blue"): 0.0,
    ("blue", "blue"): None
}
_TAGS_SIMS = {
    ("x", "x"): 1.0,
    ("x", "y"): 0.5,
    ("y", "y"): 1.0
}
TRANSFORMERS = [
    "linear_inverse",
    "exponential_inverse",
    "gaussian_inverse",
    "polynomial_inverse",
    "square_inverse"
]

EXPECTED = {}
for t in TRANSFORMERS:
    EXPECTED[t] = {
        "color.json": _distances(_COLOR_SIMS, t),
        "tags.json": _distances(_TAGS_SIMS, t),
    }
