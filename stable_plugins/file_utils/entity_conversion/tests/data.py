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

TEST_DATA = {
    "attribute_metadata.json": (
        r'[{"ID":"number","type":"integer","title":"number","description":"",'
        r'"multiple":false,"ordered":false,"separator":";","refTarget":null,"schema":null},'
        r'{"ID":"tags","type":"string","title":"tags","description":"",'
        r'"multiple":true,"ordered":true,"separator":";","refTarget":null,"schema":null}]'
    ),
    "entities.csv": '''"ID","href","number","tags"
"e1","","1","x;y"
"e2","","2",""''',
    "entities.json": (
        r'[{"ID":"e1","href":"","number":1,"tags":["x","y"]},'
        r'{"ID":"e2","href":"","number":2,"tags":[]}]'
    ),
    "vector.csv": '''"ID","href","dim0","dim1"
"v1","","0.5","1"
"v2","","2","3.25"''',
}

EXPECTED_ENTITIES = [
    {"ID": "e1", "href": "", "number": 1, "tags": ["x", "y"]},
    {"ID": "e2", "href": "", "number": 2, "tags": []},
]

EXPECTED_CSV_ROWS = [
    ["ID", "href", "number", "tags"],
    ["e1", "", "1", "x;y"],
    ["e2", "", "2", ""],
]

EXPECTED_VECTOR_ENTITIES = [
    {"ID": "v1", "href": "", "dim0": 0.5, "dim1": 1.0},
    {"ID": "v2", "href": "", "dim0": 2.0, "dim1": 3.25},
]
