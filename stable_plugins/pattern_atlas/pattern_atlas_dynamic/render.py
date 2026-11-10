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

import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from typing import Match

from flask import current_app, url_for
from jinja2 import Environment, select_autoescape
from markupsafe import Markup, escape
from mistune import create_markdown

from ..plugin import PA_BLP

from .model import (
    ImplementationPackage,
    PatternAtlasContent,
    Pattern,
    PatternLanguage,
    PatternAtlasIndex,
    QCAtlasContent,
)

CATEGORY_HEADLINES = {
    "dataencodings": "Data Encodings - How can classical data be encoded into quantum states for computation?",
    "unitarytransformations": "Unitary Transformations - How can unitary transformations be designed and composed to build quantum algorithms?",
    "warmstaring": "Warm-Starting - How can prior knowledge or favorable initializations be used to improve the convergence of quantum algorithms?",
    "programflow": "Program Flow - How can computations be effectively distributed between quantum and classical hardware?",
    "circuitcutting": "Circuit Cutting - How can large quantum circuits be partitioned into smaller, executable subcircuits to overcome hardware limitations?",
    "errorhandling": "Error Handling - How can quantum algorithms be made robust against noise and errors in current quantum hardware?",
    "quantumstates": "Quantum States - What are quantum states, and how are they represented and manipulated in quantum algorithms?",
    "execution": "Execution - What best practices should be followed when developing hybrid quantum-classical applications?",
    "development": "Development - What best practices should be followed when developing hybrid quantum-classical applications?",
    "operations": "Operations - How can the execution, monitoring, and management of quantum applications be organized and automated?",
    "measurement": "Measurement - How can classical information be accurately extracted from quantum states after computation?",
    "qml": "Quantum Machine Learning - How to use quantum computing to solve machine learning problems?",
}

# regex for latex math
BLOCK_MATH_RE = re.compile(
    r"(?<!\\)\$\$(.+?)(?<!\\)\$\$|(?<!\\)\\\[(.+?)(?<!\\)\\\]", re.S
)
INLINE_MATH_RE = re.compile(
    r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$|(?<!\\)\\\((.+?)(?<!\\)\\\)", re.S
)
LINK_IN_TEX = re.compile(r"\[([^\]]+)\]\([^)]+\)")

_CAMEL_CASE_REGEX = re.compile(r"([a-zäöü])([A-ZÄÖÜ])")


def _camel_case_replacer(match: re.Match) -> str:
    return f"{match[1]} {match[2]}".lower()


def split_camel_case(text: str) -> str:
    return _CAMEL_CASE_REGEX.sub(_camel_case_replacer, text)


# convert Markdown links [label](url) inside Latex code into plain text: \text{label}
def _link_as_text(tex: str) -> str:
    return LINK_IN_TEX.sub(r"\\text{\1}", tex)


def compatible_markdown(markdown_text: str) -> str:
    if not markdown_text:
        return "–"

    # placeholders will store all math snippets (both inline and block) from markdown_text as html
    placeholders: list[str] = []
    counter = 0

    def replace_inline(match_obj: Match[str]) -> str:
        assert INLINE_MATH_RE.fullmatch(match_obj.group(0))
        nonlocal counter
        tex = _link_as_text((match_obj.group(1) or match_obj.group(2)).strip())
        placeholders.append(f'<span class="math">\\({escape(tex)}\\)</span>')
        # the counter in the token corresponds to the index of this snippet in placeholders
        token = f"§INLINE_MATH_{counter}§"
        counter += 1
        return token

    def replace_block(match_obj: Match[str]) -> str:
        assert BLOCK_MATH_RE.fullmatch(match_obj.group(0))
        nonlocal counter
        tex = _link_as_text((match_obj.group(1) or match_obj.group(2)).strip())
        placeholders.append(f'<div class="math">\\[{escape(tex)}\\]</div>')
        # the counter in the token corresponds to the index of this snippet in placeholders
        token = f"§BLOCK_MATH_{counter}§"
        counter += 1
        return token

    # replace all math snippets with tokens so that mistune does not alter or escape them
    text = BLOCK_MATH_RE.sub(replace_block, markdown_text)
    text = INLINE_MATH_RE.sub(replace_inline, text)

    md = create_markdown(
        escape=True, plugins=["table", "footnotes", "url", "task_lists"]
    )
    html = md(text)

    # replace all tokens with their corresponding math html snippets
    for i, snippet in enumerate(placeholders):
        html = html.replace(f"§INLINE_MATH_{i}§", snippet)
        html = html.replace(f"§BLOCK_MATH_{i}§", snippet)

    return html


class ExtractImageLinksParser(HTMLParser):
    def __init__(self, *, convert_charrefs: bool = True) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        self.image_links = set()

    def reset(self) -> None:
        self.image_links = set()
        return super().reset()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            for attr, value in attrs:
                if attr == "src":
                    self.image_links.add(value)


def ui_base_url() -> str:
    index_url = url_for(f"{PA_BLP.name}.IndexUI")
    return index_url.removesuffix("/index.html")


def static_url(filename: str) -> str:
    return url_for(f"{PA_BLP.name}.static", filename=filename)


def resource(url: str | None) -> str:
    if not url:
        return static_url("empty.svg")
    return url


class DynamicRender:
    def __init__(self) -> None:
        self._jinja_envs: dict[int, Environment] = {}
        self._atlas_index = PatternAtlasIndex()

    def _get_jinja_env(self) -> Environment:
        app = current_app._get_current_object()
        env = self._jinja_envs.get(id(app))
        if env is None:
            env = app.jinja_env.overlay(
                loader=PA_BLP.jinja_loader,
                autoescape=select_autoescape(
                    enabled_extensions=("html", "jinja2"), default_for_string=True
                ),
            )
            env.filters = dict(env.filters)
            env.filters["resource"] = resource
            env.filters["markdown"] = self._markdown
            env.filters["split_camel_case"] = split_camel_case
            env.filters["json_dumps"] = json.dumps
            self._jinja_envs[id(app)] = env
        return env

    def _render(self, template_name: str, **context) -> str:
        template = self._get_jinja_env().get_template(template_name)
        return template.render(base_url=ui_base_url(), static_url=static_url, **context)

    def _markdown(self, markdown: str):
        if not markdown:
            return "–"
        html = compatible_markdown((markdown.strip()))
        assert isinstance(html, str)
        html = html.strip()
        if html.startswith("<p>") and html.endswith("</p>"):
            html = html[3:-4]

        image_link_parser = ExtractImageLinksParser()
        image_link_parser.feed(html)
        for link in image_link_parser.image_links:
            replacement = resource(link)
            html = html.replace(link, replacement)

        html = html.replace(
            'href="pattern-languages/', f'href="{ui_base_url()}/pattern-languages/'
        )
        html = html.replace('href="http', 'target="_blank" href="http')
        return Markup(html)

    def render_index(self, atlas: PatternAtlasContent) -> str:
        return self._render("languages.jinja2", atlas=atlas)

    def render_language_overview(
        self, atlas: PatternAtlasContent, language: PatternLanguage
    ) -> str:
        all_patterns = language.get_patterns_sorted(atlas)
        return self._render(
            "language-overview.jinja2",
            patterns=all_patterns,
            language=language,
            sort_label="Name: A → Z",
        )

    def render_language_overview_categorized(
        self, atlas: PatternAtlasContent, language: PatternLanguage
    ) -> str:
        all_patterns = language.get_patterns_sorted(atlas)
        patterns_by_category = defaultdict(list)
        for pattern in all_patterns:
            for category in pattern.categories:
                patterns_by_category[CATEGORY_HEADLINES[category]].append(pattern)
        return self._render(
            "language-overview.jinja2",
            patterns_by_category=patterns_by_category,
            language=language,
            sort_label="Tags",
        )

    def render_language_overview_reverse(
        self, atlas: PatternAtlasContent, language: PatternLanguage
    ) -> str:
        all_patterns = language.get_patterns_sorted(atlas)
        all_patterns.reverse()
        return self._render(
            "language-overview.jinja2",
            patterns=all_patterns,
            language=language,
            sort_label="Name: Z → A",
        )

    def render_pattern_graph(
        self, atlas: PatternAtlasContent, language: PatternLanguage
    ) -> str:
        self._atlas_index.build(atlas)
        lang_relations = self._atlas_index.relations_for_language(language.language_id)
        all_patterns = language.get_patterns_sorted(atlas)
        return self._render(
            "pattern-graph.jinja2",
            relations=lang_relations,
            patterns=all_patterns,
            language=language,
        )

    def render_pattern(
        self,
        atlas: PatternAtlasContent,
        qc_atlas: QCAtlasContent,
        pattern: Pattern,
        language: PatternLanguage,
    ) -> str:
        return self._render(
            "pattern.jinja2",
            atlas=atlas,
            qc_atlas=qc_atlas,
            pattern=pattern,
            language=language,
        )
