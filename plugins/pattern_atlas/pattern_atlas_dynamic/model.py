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

from dataclasses import dataclass, field
from typing import Union, Literal


@dataclass
class PatternLanguage:
    language_id: str
    name: str
    logo: str
    copyright_notice: str
    patterns: set[str] = field(default_factory=set)

    def get_patterns_sorted(self, atlas: "AtlasContent") -> list["Pattern"]:
        patterns = [atlas.patterns[p] for p in self.patterns]
        patterns.sort(key=lambda p: (p.name, p.pattern_id))
        return patterns


@dataclass
class Pattern:
    pattern_id: str
    pattern_language: str
    name: str
    icon: str
    citation: str
    aliases: str = ""
    tags: set[str] = field(default_factory=set)
    intent: str = ""
    intent_type: Literal["intent", "problem", "question"] = "intent"
    context: str = ""
    forces: str = ""
    solution: str = ""
    result: str = ""
    examples: str = ""
    related_patterns: str = ""
    known_uses: str = ""
    extra_sections: dict[str, str] = field(default_factory=dict)
    outgoing_edges: set[str] = field(default_factory=set)
    incoming_edges: set[str] = field(default_factory=set)
    undirected_edges: set[str] = field(default_factory=set)

    @property
    def has_related_patterns(self):
        return (
            bool(self.related_patterns)
            or bool(self.outgoing_edges)
            or bool(self.incoming_edges)
            or bool(self.undirected_edges)
        )

    def get_relations_sorted(self, atlas: "AtlasContent") -> list["PatternRelation"]:
        relations: list[PatternRelation] = []

        relations += sorted(
            (atlas.relations[r] for r in self.outgoing_edges),
            key=lambda r: r.get_target(atlas).name,
        )

        relations += sorted(
            (
                atlas.relations[r].from_persepective(self.pattern_id)
                for r in self.undirected_edges
            ),
            key=lambda r: r.get_target(atlas).name,
        )

        relations += sorted(
            (atlas.relations[r] for r in self.incoming_edges),
            key=lambda r: r.get_source(atlas).name,
        )

        return relations

    def update(self, pattern: "Pattern"):
        if self.pattern_id != pattern.pattern_id:
            raise ValueError("Can only update patterns when the pattern id matches!")

        for attr in (
            "pattern_language",
            "name",
            "icon",
            "citation",
            "aliases",
            "intent",
            "context",
            "forces",
            "solution",
            "result",
            "examples",
            "related_patterns",
            "known_uses",
        ):
            self_value = getattr(self, attr)
            if not self_value:
                setattr(self, attr, getattr(pattern, attr))

        for set_attr in (
            "tags",
            "outgoing_edges",
            "incoming_edges",
            "undirected_edges",
        ):
            self_value: set = getattr(self, set_attr)
            self_value.update(getattr(pattern, set_attr))

        self.extra_sections.update(pattern.extra_sections)


@dataclass
class PatternRelation:
    edge_id: str
    source_pattern: str
    target_pattern: str
    is_directed: bool = True
    edge_type: str = "uses"
    description: str = ""

    def get_source(self, atlas: "AtlasContent") -> Pattern:
        return atlas.patterns[self.source_pattern]

    def get_target(self, atlas: "AtlasContent") -> Pattern:
        return atlas.patterns[self.target_pattern]

    def from_persepective(self, pattern_id: str) -> "PatternRelation":
        if self.is_directed:
            raise ValueError("Cannot switch source and target for directed edges!")
        if self.source_pattern == pattern_id:
            return self
        if self.target_pattern == pattern_id:
            return PatternRelation(
                edge_id=self.edge_id,
                source_pattern=self.target_pattern,
                target_pattern=self.source_pattern,
                is_directed=False,
                edge_type=self.edge_type,
                description=self.description,
            )
        raise ValueError(
            f"Pattern {pattern_id} is not connected by this PatternRelation!"
        )


@dataclass
class AtlasContent:
    asset_map: dict[str, str] = field(default_factory=dict)
    languages: dict[str, PatternLanguage] = field(default_factory=dict)
    patterns: dict[str, Pattern] = field(default_factory=dict)
    relations: dict[str, PatternRelation] = field(default_factory=dict)

    def add(self, obj: Union[PatternLanguage, Pattern, PatternRelation]):
        if isinstance(obj, PatternLanguage):
            self.add_language(obj)
        elif isinstance(obj, Pattern):
            self.add_pattern(obj)
        elif isinstance(obj, PatternRelation):
            self.add_pattern_relation(obj)
        else:
            raise TypeError(f"Unsupported Type {type(obj)}!")

    def add_language(self, language: PatternLanguage):
        if language.language_id in self.languages:
            return
        self.languages[language.language_id] = language

    def add_pattern(self, pattern: Pattern):
        existing_pattern = self.patterns.get(pattern.pattern_id)
        if existing_pattern:
            existing_pattern.update(pattern)
        else:
            self.patterns[pattern.pattern_id] = pattern
        if pattern.pattern_language:
            language = self.languages.get(pattern.pattern_language)
            if not language:
                raise KeyError(
                    "Attempted to add Pattern before adding its Pattern Language!"
                )
            language.patterns.add(pattern.pattern_id)

    def add_pattern_relation(self, pattern_relation: PatternRelation):
        if pattern_relation.edge_id in self.relations:
            return
        self.relations[pattern_relation.edge_id] = pattern_relation
        source_pattern = self.patterns.get(pattern_relation.source_pattern)
        target_pattern = self.patterns.get(pattern_relation.target_pattern)
        if not source_pattern:
            raise KeyError("Attempted to add Relation before its source Pattern!")
        if not target_pattern:
            raise KeyError("Attempted to add Relation before its target Pattern!")
        if pattern_relation.is_directed:
            source_pattern.outgoing_edges.add(pattern_relation.edge_id)
            target_pattern.incoming_edges.add(pattern_relation.edge_id)
        else:
            source_pattern.undirected_edges.add(pattern_relation.edge_id)
            target_pattern.undirected_edges.add(pattern_relation.edge_id)
