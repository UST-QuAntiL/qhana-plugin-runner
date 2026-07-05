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

"""Test data for the attribute distance aggregator plugin.

Three entities with a single value attribute ("color") and a multi value
attribute ("tags", empty for e3). The element distances for "color" contain a
null distance for the (blue, blue) pair to exercise the missing data handling.
"""

import json as _json

TEST_DATA = {
    "attribute_metadata.json": r'[{"ID":"color","type":"color","title":"","description":"ref","multiple":false,"ordered":false,"separator":";","refTarget":"taxonomies.zip:color.json","schema":null},{"ID":"tags","type":"tags","title":"","description":"ref","multiple":true,"ordered":false,"separator":";","refTarget":"taxonomies.zip:tags.json","schema":null}]',
    "entities.json": r'[{"ID":"e1","color":"red","tags":["x","y"]},{"ID":"e2","color":"blue","tags":["y"]},{"ID":"e3","color":"red","tags":[]}]',
    "entities.csv": '''"ID","color","tags"
"e1","red","x;y"
"e2","blue","y"
"e3","red",""''',
    "element-distances": {
        "color.json": r'[{"source":"red","target":"red","distance":0.0},{"source":"red","target":"blue","distance":1.0},{"source":"blue","target":"blue","distance":null}]',
        "tags.json": r'[{"source":"x","target":"x","distance":0.0},{"source":"x","target":"y","distance":0.6},{"source":"y","target":"y","distance":0.0}]',
    },
}

TEST_DATA["entities_lines.json"] = "\n".join(
    _json.dumps(entity) for entity in _json.loads(TEST_DATA["entities.json"])
)


def _distances(values):
    return [
        {"source": source, "target": target, "distance": distance}
        for (source, target), distance in values.items()
    ]


# Color is single valued, so only the missing data handling of the null
# (blue, blue) element distance differs between the scenarios: "ignore" drops
# the pair (distance null), "mean" replaces it with mean(0.0, 1.0) = 0.5 and
# "max" with 1.0. The tags element distances contain no nulls.
_COLOR_IGNORE = _distances(
    {
        ("e1", "e1"): 0.0,
        ("e1", "e2"): 1.0,
        ("e1", "e3"): 0.0,
        ("e2", "e2"): None,
        ("e2", "e3"): 1.0,
        ("e3", "e3"): 0.0,
    }
)


# Tags distance lists per entity pair (pairs with e3 have an empty cross
# product and therefore a null distance):
# (e1, e1): [0.0, 0.6, 0.6, 0.0], (e1, e2): [0.6, 0.0], (e2, e2): [0.0]
def _tags_expected(e1_e1, e1_e2, e2_e2):
    return _distances(
        {
            ("e1", "e1"): e1_e1,
            ("e1", "e2"): e1_e2,
            ("e1", "e3"): None,
            ("e2", "e2"): e2_e2,
            ("e2", "e3"): None,
            ("e3", "e3"): None,
        }
    )


EXPECTED = {
    ("mean", "ignore"): {
        "color.json": _COLOR_IGNORE,
        "tags.json": _tags_expected(0.3, 0.3, 0.0),
    },
    ("median", "ignore"): {
        "color.json": _COLOR_IGNORE,
        "tags.json": _tags_expected(0.3, 0.3, 0.0),
    },
    ("max", "ignore"): {
        "color.json": _COLOR_IGNORE,
        "tags.json": _tags_expected(0.6, 0.6, 0.0),
    },
    ("min", "ignore"): {
        "color.json": _COLOR_IGNORE,
        "tags.json": _tags_expected(0.0, 0.0, 0.0),
    },
    ("mean", "mean"): {
        "color.json": _distances(
            {
                ("e1", "e1"): 0.0,
                ("e1", "e2"): 1.0,
                ("e1", "e3"): 0.0,
                ("e2", "e2"): 0.5,
                ("e2", "e3"): 1.0,
                ("e3", "e3"): 0.0,
            }
        ),
        "tags.json": _tags_expected(0.3, 0.3, 0.0),
    },
    ("mean", "max"): {
        "color.json": _distances(
            {
                ("e1", "e1"): 0.0,
                ("e1", "e2"): 1.0,
                ("e1", "e3"): 0.0,
                ("e2", "e2"): 1.0,
                ("e2", "e3"): 1.0,
                ("e3", "e3"): 0.0,
            }
        ),
        "tags.json": _tags_expected(0.3, 0.3, 0.0),
    },
}
