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

from .model import AtlasContent, Pattern, PatternLanguage

BLOCK_MATH_RE  = re.compile(r'(?<!\\)\$\$(.+?)(?<!\\)\$\$', re.S)
INLINE_MATH_RE = re.compile(r'(?<!\\)\$(?!\$)(.+?)(?<!\\)\$', re.S)
LINK_IN_TEX    = re.compile(r'\[([^\]]+)\]\([^)]+\)')

_CAMEL_CASE_REGEX = re.compile(r"([a-zäöü])([A-ZÄÖÜ])")


def _camel_case_replacer(match: re.Match) -> str:
    return f"{match[1]} {match[2]}".lower()


def split_camel_case(text: str) -> str:
    return _CAMEL_CASE_REGEX.sub(_camel_case_replacer, text)

def _sanitize_tex(tex: str) -> str:
    return LINK_IN_TEX.sub(r'\\text{\1}', tex)

def compat_markdown(md_text: str) -> str:
    if not md_text:
        return "–"
    
    placeholders: list[str] = []
    counter = 0

    def repl_inline(m: Match[str]) -> str:
        nonlocal counter
        tex = _sanitize_tex(m.group(1).strip())
        placeholders.append(f'<span class="math">\\({tex}\\)</span>')
        token = f"§INLINE_MATH_{counter}§"
        counter += 1
        return token

    def repl_block(m: Match[str]) -> str:
        nonlocal counter
        tex = _sanitize_tex(m.group(1).strip())
        placeholders.append(f'<div class="math">\\[{tex}\\]</div>')
        token = f"§BLOCK_MATH_{counter}§"
        counter += 1
        return token
    
    text = BLOCK_MATH_RE.sub(repl_block, md_text)
    text = INLINE_MATH_RE.sub(repl_inline, text)

    md = create_markdown(escape=False, plugins=["table", "footnotes", "url", "task_lists"])
    html = md(text)

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
        self._mistune = create_markdown(
            escape=False, plugins=["table", "footnotes", "url", "task_lists"]
        )

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
        print("RAW_MARKDOWN_INPUT:", repr(markdown))
        html = compat_markdown((markdown.strip()))
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
        return template.render(atlas=atlas, base_url=f"/plugins/{PA_BLP.name}/ui", is_planqk=self.is_planqk)

    def render_language_overview(self, atlas: AtlasContent, language: PatternLanguage, query: str) -> str:
        def matches(pattern: Pattern, query: str) -> bool:
            name_ok = query in (pattern.name or "").lower()
            tags_ok = any(query in (t or "").lower() for t in (pattern.tags or []))
            return name_ok or tags_ok
        
        template = self._jinja.get_template("language-overview.jinja2")
        all_patterns = language.get_patterns_sorted(atlas)
        if query:
            filtered = [p for p in all_patterns if matches(p, query)]
        else:
            filtered = all_patterns        
        return template.render(patterns=filtered, base_url=f"/plugins/{PA_BLP.name}/ui", language=language, is_planqk=self.is_planqk)

    def render_pattern(self, atlas: AtlasContent, pattern: Pattern, language: PatternLanguage) -> str:
        template = self._jinja.get_template("pattern.jinja2")
        return template.render(atlas=atlas, base_url=f"/plugins/{PA_BLP.name}/ui", pattern=pattern, language=language, is_planqk=self.is_planqk)

