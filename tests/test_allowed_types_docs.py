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

"""
Tests that the allowed types documentation matches the allowed types definition.

``docs/data-formats/allowed-types.md`` is the documentation of the enums in
``tests/allowed_types.py``. Both are maintained by hand, so this module fails
whenever they drift apart or whenever the anchors the plugin documentation links
to would break.
"""

import importlib.util
import re
from pathlib import Path

import pytest

from .allowed_types import (
    AllowedContentTypes,
    AllowedDataTypesNoFormat,
    AllowedDataTypesWithFormat,
)

REPO_ROOT = Path(__file__).parents[1]
ALLOWED_TYPES_DOC = REPO_ROOT / "docs" / "data-formats" / "allowed-types.md"
PLUGIN_AUTODOC = REPO_ROOT / "docs" / "plugin_autodoc.py"

# An explicit MyST target followed by the definition list term naming the type, e.g.
#     (dt-entity-vector)=
#     `entity/vector`
_ENTRY = re.compile(r"^\((dt|ct)-(\S+?)\)=\n`([^`\n]+)`$", re.MULTILINE)


def _documented_types() -> dict[str, dict[str, str]]:
    """Map ``dt``/``ct`` to the documented types and their anchors."""
    doc = ALLOWED_TYPES_DOC.read_text(encoding="utf-8")
    documented: dict[str, dict[str, str]] = {"dt": {}, "ct": {}}
    for prefix, slug, value in _ENTRY.findall(doc):
        assert (
            value not in documented[prefix]
        ), f"{value} is documented more than once as a {prefix} entry"
        documented[prefix][value] = f"{prefix}-{slug}"
    return documented


def _load_plugin_autodoc():
    """Import ``docs/plugin_autodoc.py`` by path (``docs`` is not a package)."""
    spec = importlib.util.spec_from_file_location("plugin_autodoc", PLUGIN_AUTODOC)
    assert spec and spec.loader, f"Could not load {PLUGIN_AUTODOC}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _allowed_data_types() -> set[str]:
    return {e.value for e in AllowedDataTypesWithFormat} | {
        e.value for e in AllowedDataTypesNoFormat
    }


def _allowed_content_types() -> set[str]:
    return {e.value for e in AllowedContentTypes}


def _assert_same(documented: set[str], allowed: set[str], kind: str):
    missing = sorted(allowed - documented)
    extra = sorted(documented - allowed)
    if missing or extra:
        pytest.fail(
            f"{ALLOWED_TYPES_DOC.relative_to(REPO_ROOT)} is out of sync with the {kind}"
            f" enums in tests/allowed_types.py:\n"
            f"  allowed but not documented: {missing or 'none'}\n"
            f"  documented but not allowed: {extra or 'none'}"
        )


def _type_anchor(value: str, prefix: str) -> str:
    # "*" would slugify to nothing, so it is spelled out ("*" -> "wildcard",
    # "entity/*" -> "entity-wildcard")
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower().replace("*", "wildcard")).strip("-")
    return f"{prefix}-{slug}"


@pytest.mark.parametrize("prefix", ["dt", "ct"])
def test_documentation_is_not_empty(prefix):
    documented = _documented_types()[prefix]
    assert documented, (
        f"No '{prefix}' entries found in {ALLOWED_TYPES_DOC.relative_to(REPO_ROOT)}."
        " Either the file lost its entries or the parser no longer matches its format."
    )


def test_documented_data_types_match_allowed_data_types():
    _assert_same(set(_documented_types()["dt"]), _allowed_data_types(), "data type")


def test_documented_content_types_match_allowed_content_types():
    _assert_same(set(_documented_types()["ct"]), _allowed_content_types(), "content type")


def test_documented_anchors_are_the_generated_anchors():
    """The overview in ``docs/all-plugins.md`` links to these anchors, so they must match."""
    documented = _documented_types()
    for prefix, values in (
        ("dt", _allowed_data_types()),
        ("ct", _allowed_content_types()),
    ):
        for value in sorted(values):
            expected = _type_anchor(value, prefix)
            actual = documented[prefix].get(value)
            assert actual == expected, (
                f"Anchor for '{value}' is '({actual})=' but the generated links in"
                f" docs/all-plugins.md point at '#{expected}'."
            )


def test_plugin_autodoc_anchors_agree():
    """``plugin_autodoc.py`` keeps its own copy of ``type_anchor``."""
    plugin_autodoc = _load_plugin_autodoc()
    for prefix, values in (
        ("dt", _allowed_data_types()),
        ("ct", _allowed_content_types()),
    ):
        for value in sorted(values):
            assert plugin_autodoc.type_anchor(value, prefix) == _type_anchor(
                value, prefix
            ), (
                f"docs/plugin_autodoc.py and tests/allowed_types.py disagree on the anchor"
                f" for '{value}'."
            )
