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

import json as _json

TEST_DATA = {
    "attribute_metadata.json": r'[{"ID":"color","type":"color","title":"","description":"ref","multiple":false,"ordered":false,"separator":";","refTarget":"taxonomies.zip:color.json","schema":null},{"ID":"tags","type":"tags","title":"","description":"ref","multiple":true,"ordered":false,"separator":";","refTarget":"taxonomies.zip:tags.json","schema":null}]',
    "entities.json": r'[{"ID":"e1","color":"red","tags":["x","y"]},{"ID":"e2","color":"blue","tags":["y"]},{"ID":"e3","tags":[]}]',
    "entities.csv": '''"ID","color","tags"
"e1","red","x;y"
"e2","blue","y"
"e3","",""''',
    "element-distances": {
        "color.json": r'[{"source":"red","target":"red","distance":0.0},{"source":"red","target":"blue","distance":1.0},{"source":"blue","target":"blue","distance":0.0}]',
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


EXPECTED = {
    "color.json": _distances(
        {
            ("e1", "e2"): 1.0,
            ("e1", "e3"): None,
            ("e2", "e3"): None,
        }
    ),
    "tags.json": _distances(
        {
            ("e1", "e2"): 0.15,
            ("e1", "e3"): None,
            ("e2", "e3"): None,
        }
    ),
}
