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

from typing import Literal

import pytest

from muse_for_music.util import (
    _parse_taxonomy_mapping,
    taxonomy_to_entity,
)


def _build_taxonomy(
    items, na_item=None, kind: Literal["tree", "list"] = "tree", name="Test"
):
    entity = {
        "_links": {"self": {"href": f"http://localhost/api/taxonomies/list/{name}"}},
        "taxonomy_type": kind,
        "items": items,
    }
    if na_item is not None:
        entity["na_item"] = na_item
    return entity


def _node(id_, name, mapping=None, children=None):
    """Build a tree node, omitting ``mapping`` when not provided."""
    node = {"id": id_, "name": name, "children": children or []}
    if mapping is not None:
        node["mapping"] = mapping
    return node


def _by_id(entities):
    return {e["ID"]: e for e in entities if isinstance(e, dict)}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("1,2,3", [1.0, 2.0, 3.0], id="basic"),
        pytest.param("  1 ,  2.5 , 3 ", [1.0, 2.5, 3.0], id="strips-whitespace"),
        pytest.param("", [], id="empty-string"),
        pytest.param("   ", [], id="blank-string"),
        pytest.param("1,,2,", [1.0, 2.0], id="skips-empty-tokens"),
    ],
)
def test_parse_mapping(raw, expected):
    assert _parse_taxonomy_mapping(raw) == expected


def test_parse_mapping_bad_token_raises():
    with pytest.raises(ValueError, match="bad token at index 1: 'x'"):
        _parse_taxonomy_mapping("1,x,3")


def test_tree_mapping_parsed_and_raw_kept():
    root = _node(1, "root", mapping="1,2,3", children=[])
    result = taxonomy_to_entity(_build_taxonomy(root))

    entities = _by_id(result.entities)
    assert entities["t_Test_1"]["mapping"] == [1.0, 2.0, 3.0]
    assert entities["t_Test_1"]["mapping_raw"] == "1,2,3"


def test_tree_mapping_padded_to_common_dimension():
    root = _node(
        1,
        "root",
        mapping="1,2",
        children=[_node(2, "child", mapping="3,4,5")],
    )
    result = taxonomy_to_entity(_build_taxonomy(root))

    entities = _by_id(result.entities)
    # widest mapping has 3 entries, so the shorter one is zero padded
    assert entities["t_Test_1"]["mapping"] == [1.0, 2.0, 0]
    assert entities["t_Test_2"]["mapping"] == [3.0, 4.0, 5.0]
    # padding does not alter the preserved raw string
    assert entities["t_Test_1"]["mapping_raw"] == "1,2"


def test_tree_missing_mapping_defaults_to_empty_then_padded():
    root = _node(
        1,
        "root",  # no mapping key at all
        children=[_node(2, "child", mapping="7,8")],
    )
    result = taxonomy_to_entity(_build_taxonomy(root))

    entities = _by_id(result.entities)
    assert entities["t_Test_1"]["mapping"] == [0, 0]
    assert entities["t_Test_1"]["mapping_raw"] == ""
    assert entities["t_Test_2"]["mapping"] == [7.0, 8.0]


def test_tree_no_mappings_leaves_empty_lists_unpadded():
    root = _node(1, "root", children=[_node(2, "child")])
    result = taxonomy_to_entity(_build_taxonomy(root))

    entities = _by_id(result.entities)
    # mapping_dimension is 0, so no padding happens at all
    assert entities["t_Test_1"]["mapping"] == []
    assert entities["t_Test_2"]["mapping"] == []


def test_tree_na_item_mapping_parsed_and_padded():
    root = _node(1, "root", mapping="1,2,3", children=[])
    na_item = {"id": 99, "name": "na", "mapping": "9"}
    result = taxonomy_to_entity(_build_taxonomy(root, na_item=na_item))

    entities = _by_id(result.entities)
    na = entities["t_Test_na"]
    assert na["tax_item_name"] == "na"
    assert na["mapping"] == [9.0, 0, 0]
    assert na["mapping_raw"] == "9"
    # na is linked under the assumed root node
    assert {"source": "t_Test_1", "target": "t_Test_na"} in result.relations


def test_tree_bad_mapping_token_raises():
    root = _node(1, "root", mapping="1,bad,3", children=[])
    with pytest.raises(ValueError, match="bad token at index 1"):
        taxonomy_to_entity(_build_taxonomy(root))


def test_list_mapping_parsed_and_padded():
    items = [
        {"id": 1, "name": "a", "mapping": "1.5"},
        {"id": 2, "name": "b", "mapping": "2.5,3.5"},
    ]
    result = taxonomy_to_entity(_build_taxonomy(items, kind="list"))

    entities = _by_id(result.entities)
    assert entities["t_Test_root"]["tax_item_name"] == "root"
    assert entities["t_Test_root"]["mapping"] == [0, 0]
    assert entities["t_Test_1"]["mapping"] == [1.5, 0]
    assert entities["t_Test_2"]["mapping"] == [2.5, 3.5]


def test_list_na_item_mapping_parsed():
    items = [{"id": 1, "name": "a", "mapping": "1,2"}]
    na_item = {"id": 5, "name": "na", "mapping": "4"}
    result = taxonomy_to_entity(_build_taxonomy(items, na_item=na_item, kind="list"))

    entities = _by_id(result.entities)
    assert entities["t_Test_na"]["mapping"] == [4.0, 0]
    assert {"source": "t_Test_root", "target": "t_Test_na"} in result.relations


def test_unknown_taxonomy_type_raises():
    entity = {
        "_links": {"self": {"href": "http://localhost/api/taxonomies/tree/Test"}},
        "taxonomy_type": "graph",
        "items": [],
    }
    with pytest.raises(ValueError, match="Unknown taxonomy type graph"):
        taxonomy_to_entity(entity)
