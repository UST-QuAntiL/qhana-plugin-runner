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

from collections import defaultdict
import re
from hashlib import sha3_256
from html.parser import HTMLParser
from pathlib import Path
from json import dumps, loads

from httpx import get
from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup
from mistune import create_markdown
from typing import Match
from pattern_atlas.plugin import PA_BLP

from .model import AtlasContent, Pattern, PatternLanguage, AtlasIndex

CATEGORY_HEADLINES = {
    "dataencodings": "Data Encodings- How can classical data be encoded into quantum states for computation?",
    "unitarytransformations": "Unitary Transformations - How can unitary transformations be designed and composed to build quantum algorithms?",
    "warmstaring": "Warm-Starting- How can prior knowledge or favorable initializations be used to improve the convergence of quantum algorithms?",
    "programflow": "Program Flow- How can computations be effectively distributed between quantum and classical hardware?",
    "circuitcutting": "Circuit Cutting- How can large quantum circuits be partitioned into smaller, executable subcircuits to overcome hardware limitations?",
    "errorhandling": "Error Handling- How can quantum algorithms be made robust against noise and errors in current quantum hardware?",
    "quantumstates": "Quantum States- What are quantum states, and how are they represented and manipulated in quantum algorithms?",
    "execution": "Execution- What best practices should be followed when developing hybrid quantum-classical applications?",
    "development": "Development- What best practices should be followed when developing hybrid quantum-classical applications?",
    "operations": "Operations- How can the execution, monitoring, and management of quantum applications be organized and automated?",
    "measurement": "Measurement- How can classical information be accurately extracted from quantum states after computation?",
    "qml": "Quantum Machine Learning- How to use quantum computing to solve machine learning problems?",
}

# regex for latex math
BLOCK_MATH_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.S)
INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", re.S)
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
        tex = _link_as_text(match_obj.group(1).strip())
        placeholders.append(f'<span class="math">\\({tex}\\)</span>')
        # the counter in the token corresponds to the index of this snippet in placeholders
        token = f"§INLINE_MATH_{counter}§"
        counter += 1
        return token

    def replace_block(match_obj: Match[str]) -> str:
        assert BLOCK_MATH_RE.fullmatch(match_obj.group(0))
        nonlocal counter
        tex = _link_as_text(match_obj.group(1).strip())
        placeholders.append(f'<div class="math">\\[{tex}\\]</div>')
        # the counter in the token corresponds to the index of this snippet in placeholders
        token = f"§BLOCK_MATH_{counter}§"
        counter += 1
        return token

    # replace all math snippets with tokens so that mistune does not alter or escape them
    text = BLOCK_MATH_RE.sub(replace_block, markdown_text)
    text = INLINE_MATH_RE.sub(replace_inline, text)

    md = create_markdown(
        escape=False, plugins=["table", "footnotes", "url", "task_lists"]
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


class DynamicRender:

    def __init__(self, is_planqk: bool = False) -> None:
        self.is_planqk = is_planqk
        self._resource_map: dict[str, str] = {"": "/ui/assets/empty.svg"}
        self._resource_bytes: dict[str, bytes] = {}
        self._jinja = Environment(
            loader=PackageLoader("pattern_atlas.pattern_atlas_dynamic"),
            autoescape=select_autoescape(
                enabled_extensions=("html", "jinja2", "css"), default_for_string=True
            ),
        )
        self._jinja.filters["resource"] = self._resource
        self._jinja.filters["markdown"] = self._markdown
        self._jinja.filters["split_camel_case"] = split_camel_case
        self._atlas_index = AtlasIndex()

    def _resource(self, url: str) -> str:
        if url is None:
            url = ""

        resolved = self._resource_map.get(url)
        if resolved:
            return resolved

        try:
            response = get(
                url, headers={"Accept": "*/*"}, timeout=3, follow_redirects=True
            )
            response.raise_for_status()
            byte_content = response.content
            suffixes = Path(response.url.path).suffixes
            asset_url = f"/plugins/{PA_BLP.name}/ui/assets/{sha3_256(byte_content).hexdigest()}{''.join(suffixes)}"
            self._resource_map[url] = asset_url
            self._resource_bytes[asset_url] = byte_content
        except Exception:
            print(f"Failed to download resource: {url}")
            self._resource_map[url] = f"/plugins/{PA_BLP.name}/ui/assets/empty.svg#" + url

        return self._resource_map[url]

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
            replacement = self._resource(link)
            html = html.replace(link, replacement)

        html = html.replace('href="pattern-languages/', 'href="/pattern-languages/')
        html = html.replace('href="http', 'target="_blank" href="http')
        return Markup(html)

    def render_empty_picture_asset(self):
        template = self._jinja.get_template("empty.svg")
        svg = template.render()
        svg_bytes = svg.encode("utf-8")

        asset_url = f"/plugins/{PA_BLP.name}/ui/assets/empty.svg"
        self._resource_map[""] = asset_url
        self._resource_bytes[asset_url] = svg_bytes

    def render_styles(self) -> str:
        template = self._jinja.get_template("styles.css")
        return template.render()

    def render_index(self, atlas: AtlasContent) -> str:
        template = self._jinja.get_template("languages.jinja2")
        return template.render(
            atlas=atlas, base_url=f"/plugins/{PA_BLP.name}/ui", is_planqk=self.is_planqk
        )

    def render_language_overview(
        self, atlas: AtlasContent, language: PatternLanguage
    ) -> str:
        template = self._jinja.get_template("language-overview.jinja2")
        all_patterns = language.get_patterns_sorted(atlas)
        return template.render(
            patterns=all_patterns,
            base_url=f"/plugins/{PA_BLP.name}/ui",
            language=language,
            is_planqk=self.is_planqk,
        )

    def render_language_overview_categorized(
        self, atlas: AtlasContent, language: PatternLanguage
    ) -> str:
        template = self._jinja.get_template("language-overview-categorized.jinja2")
        all_patterns = language.get_patterns_sorted(atlas)
        patterns_by_category = defaultdict(list)
        for pattern in all_patterns:
            for category in pattern.categories:
                patterns_by_category[CATEGORY_HEADLINES[category]].append(pattern)
        return template.render(
            patterns_by_category=patterns_by_category,
            base_url=f"/plugins/{PA_BLP.name}/ui",
            language=language,
            is_planqk=self.is_planqk,
        )

    def render_language_overview_reverse(
        self, atlas: AtlasContent, language: PatternLanguage
    ) -> str:
        template = self._jinja.get_template("language-overview-reversed.jinja2")
        all_patterns = language.get_patterns_sorted(atlas)
        all_patterns.reverse()
        return template.render(
            patterns=all_patterns,
            base_url=f"/plugins/{PA_BLP.name}/ui",
            language=language,
            is_planqk=self.is_planqk,
        )
    
    def render_pattern_graph(self, atlas: AtlasContent, language: PatternLanguage) -> str:
        template = self._jinja.get_template("pattern-graph.jinja2")
        self._atlas_index.build(atlas)
        lang_relations = self._atlas_index.relations_for_language(language.language_id)
        all_patterns = language.get_patterns_sorted(atlas)
        return template.render(
            relations = lang_relations,
            patterns = all_patterns,
            base_url=f"/plugins/{PA_BLP.name}/ui",
            language = language,
            is_plank=self.is_planqk,
        )

    def render_pattern(
        self, atlas: AtlasContent, pattern: Pattern, language: PatternLanguage
    ) -> str:
        template = self._jinja.get_template("pattern.jinja2")
        return template.render(
            atlas=atlas,
            base_url=f"/plugins/{PA_BLP.name}/ui",
            pattern=pattern,
            language=language,
            is_planqk=self.is_planqk,
        )
