# Copyright 2022 QHAna plugin runner contributors.
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

"""Tests for the the template utility."""

from qhana_plugin_runner.util.templates import (
        QHanaTemplateCategory
)

def test_tags_match_expr_edge_cases():
    tags = ["a"]
    expr = "a"
    
    assert QHanaTemplateCategory.tags_match_expr(tags, expr)
    
    expr = { "notdefined": ["a", "b", "c"] }
    
    assert False == QHanaTemplateCategory.tags_match_expr(tags, expr)

def test_tags_match_expr_or():
    tags = ["a"]
    expr = {"or": [
        "a", "b", "c"
    ]}

    assert QHanaTemplateCategory.tags_match_expr(tags, expr)
    
    expr = {"or": [
        "b", "c", "aber"
    ]}

    assert False == QHanaTemplateCategory.tags_match_expr(tags, expr)

def test_tags_match_expr_and():
    tags = ["a", "b", "c"]
    expr = {"and": [
        "a", "c"
    ]}

    assert QHanaTemplateCategory.tags_match_expr(tags, expr)
    
    expr = {"and": [
        "b", "c", "d"
    ]}

    assert False == QHanaTemplateCategory.tags_match_expr(tags, expr)

def test_tags_match_expr_not():
    tags = ["a", "b", "c"]
    expr = {"not": "d"}

    assert QHanaTemplateCategory.tags_match_expr(tags, expr)
    
    expr = {"not": "a"}

    assert False == QHanaTemplateCategory.tags_match_expr(tags, expr)
    

def test_tags_match_expr_combination():
    tags = ["a", "b", "c", "d", "e"]
    expr = {"and": [
        "a", "b", "c",
        {"not": "f"},
        {"or": [
            "c", "z"
        ]}
    ]}

    assert QHanaTemplateCategory.tags_match_expr(tags, expr)
    