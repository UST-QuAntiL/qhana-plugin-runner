# Copyright 2024 University of Stuttgart
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
from urllib.parse import urljoin, urlparse

from httpx import get

from .model import AtlasContent, Pattern, PatternLanguage, PatternRelation

KNOWN_PATTERN_SECTIONS = {
    "License",
    "Icon",
    "Alias",
    "Forces",
    "Intent",
    "Problem",
    "Driving Question",
    "Result",
    "Context",
    "Example",
    "Examples",
    "Solution",
    "Sketch",
    "Solution Sketch",
    "Known Uses",
    "Related Pattern",
    "Related Patterns",
}


def _get_content(pattern_content: dict[str, str], attr: str) -> str:
    value = pattern_content.get(attr, "")
    if not value or value in (
        "–",
        "—",
        "-",
        "--",
        "---",
        "Enter your input for this section here.",
    ):
        return ""
    return value


class AtlasClient:
    def __init__(self, atlas_url: str) -> None:
        if not atlas_url.endswith("/"):
            atlas_url = atlas_url + "/"
        urlparse(atlas_url)  # ensure a valid url
        self.atlas_url = atlas_url

    def get_all(self) -> AtlasContent:
        # from example_data import EXAMPLE_DATA

        # return EXAMPLE_DATA
        atlas = AtlasContent()
        languages = self.get_pattern_languages()

        relations = []

        for lang in languages:
            atlas.add(lang)
            patterns = self.get_patterns(language=lang)
            for pattern in patterns:
                atlas.add(pattern)
            relations.extend(self.get_pattern_relations(lang))

        for rel in relations:
            atlas.add(rel)
        return atlas

    def get_pattern_languages(self) -> list[PatternLanguage]:
        url = urljoin(self.atlas_url, "./patternLanguages")
        response = get(url, headers={"Accept": "application/hal+json, application/json"})
        response.raise_for_status()
        try:
            data = response.json().get("_embedded", {}).get("patternLanguageModels", [])
        except Exception as err:
            raise ValueError(f"Could not decode json response from url '{url}'!") from err
        return [
            PatternLanguage(
                language_id=lang["id"],
                name=lang["name"],
                logo=lang["logo"],
                copyright_notice=lang["creativeCommonsReference"],
            )
            for lang in data
        ]

    def get_patterns(self, language: PatternLanguage) -> list[Pattern]:
        url = urljoin(
            self.atlas_url, f"./patternLanguages/{language.language_id}/patterns"
        )
        response = get(url, headers={"Accept": "application/hal+json, application/json"})
        response.raise_for_status()
        try:
            data = response.json().get("_embedded", {}).get("patternModels", [])
        except Exception as err:
            raise ValueError(f"Could not decode json response from url '{url}'!") from err
        patterns = [
            Pattern(
                pattern_id=pattern["id"],
                pattern_language=language.language_id,
                name=pattern["name"],
                icon=pattern["iconUrl"],
                citation="",
                tags=set([pattern["tags"]]) if pattern.get("tags") else set(),
            )
            for pattern in data
        ]
        for pattern in patterns:
            self._update_pattern(pattern)
        return patterns

    def _update_pattern(self, pattern: Pattern) -> Pattern:
        url = urljoin(
            self.atlas_url,
            f"./patternLanguages/{pattern.pattern_language}/patterns/{pattern.pattern_id}",
        )
        response = get(url, headers={"Accept": "application/hal+json, application/json"})
        response.raise_for_status()
        try:
            data = response.json()
        except Exception as err:
            raise ValueError(f"Could not decode json response from url '{url}'!") from err

        pattern.citation = data["paperRef"] if data["paperRef"] else ""

        content = data["content"]

        license_info = _get_content(content, "License")
        if license_info and pattern.citation in license_info:
            pattern.citation = license_info

        alias = _get_content(content, "Alias")
        if alias:
            pattern.aliases = alias

        forces = _get_content(content, "Forces")
        if forces:
            pattern.forces = forces

        intent = _get_content(content, "Intent")
        if intent:
            pattern.intent = intent

        for section_alias, intent_type in (
            ("Problem", "problem"),
            ("Driving Question", "question"),
        ):
            assert intent_type in ("intent", "problem", "question")
            question = _get_content(content, section_alias)
            if question:
                if not pattern.intent:
                    pattern.intent_type = intent_type
                pattern.intent = (
                    pattern.intent + "\n\n" + question if pattern.intent else question
                )

        result = _get_content(content, "Result")
        if result:
            pattern.result = result

        context = _get_content(content, "Context")
        if context:
            pattern.context = context

        for section_alias in ("Example", "Examples"):
            examples = _get_content(content, section_alias)
            if examples:
                pattern.examples = (
                    pattern.examples + "\n\n" + examples if pattern.examples else examples
                )

        solution = _get_content(content, "Solution")
        if solution:
            pattern.solution = solution

        for section in ("Sketch", "Solution Sketch"):
            sketch = _get_content(content, section)
            if sketch:
                pattern.solution = (
                    pattern.solution + "\n\n---\n\n" + sketch
                    if pattern.solution
                    else sketch
                )

        known_uses = _get_content(content, "Known Uses")
        if known_uses:
            pattern.known_uses = known_uses

        for section_alias in ("Related Pattern", "Related Patterns"):
            related_patterns = _get_content(content, section_alias)
            if related_patterns:
                pattern.related_patterns = (
                    pattern.related_patterns + "\n\n" + related_patterns
                    if pattern.related_patterns
                    else related_patterns
                )

        for section in content.keys():
            if section in KNOWN_PATTERN_SECTIONS:
                continue
            section_content = _get_content(content, section)
            if section_content:
                pattern.extra_sections[section] = section_content

        return pattern

    def get_pattern_relations(self, language: PatternLanguage) -> list[PatternRelation]:
        directed_url = urljoin(
            self.atlas_url, f"./patternLanguages/{language.language_id}/directedEdges"
        )
        response = get(
            directed_url, headers={"Accept": "application/hal+json, application/json"}
        )
        response.raise_for_status()
        try:
            directed_data = (
                response.json().get("_embedded", {}).get("directedEdgeModels", [])
            )
        except Exception as err:
            raise ValueError(
                f"Could not decode json response from url '{directed_url}'!"
            ) from err
        relations = [
            PatternRelation(
                edge_id=edge["id"],
                is_directed=True,
                edge_type=edge["type"],
                description=edge["description"] if edge["description"] else "",
                source_pattern=edge["sourcePatternId"],
                target_pattern=edge["targetPatternId"],
            )
            for edge in directed_data
        ]
        url = urljoin(
            self.atlas_url, f"./patternLanguages/{language.language_id}/undirectedEdges"
        )
        response = get(url, headers={"Accept": "application/hal+json, application/json"})
        response.raise_for_status()
        try:
            data = response.json().get("_embedded", {}).get("undirectedEdgeModels", [])
        except Exception as err:
            raise ValueError(f"Could not decode json response from url '{url}'!") from err
        relations += [
            PatternRelation(
                edge_id=edge["id"],
                is_directed=False,
                edge_type=edge["type"],
                description=edge["description"] if edge["description"] else "",
                source_pattern=edge["pattern1Id"],
                target_pattern=edge["pattern2Id"],
            )
            for edge in data
        ]
        return relations
