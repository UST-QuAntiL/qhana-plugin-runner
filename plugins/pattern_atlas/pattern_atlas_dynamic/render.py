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
from mistune.plugins import math
from mistune.util import escape

from .model import AtlasContent, Pattern, PatternLanguage

# monkeypatch math plugin to use better regexes
# https://github.com/lepture/mistune/issues/330
math.BLOCK_MATH_PATTERN = r"(?!^ {4,})\$\$\s*(?P<math_text>\S.*?)\s*(?<!\\)\$\$"
math.INLINE_MATH_PATTERN = r"\$\s*(?P<math_text>\S.*?)\s*(?<!\\)\$"

# monkeyfix mistune to correctly escape rendered math
_render_block_math = math.render_block_math
_render_inline_math = math.render_inline_math


def render_block_math_fixed(renderer, text):
    return _render_block_math(renderer, escape(text))


def render_inline_math_fixed(renderer, text):
    return _render_inline_math(renderer, escape(text))


math.render_block_math = render_block_math_fixed
math.render_inline_math = render_inline_math_fixed


_CAMEL_CASE_REGEX = re.compile(r"([a-zäöü])([A-ZÄÖÜ])")


def _camel_case_replacer(match: re.Match) -> str:
    return f"{match[1]} {match[2]}".lower()


def split_camel_case(text: str) -> str:
    return _CAMEL_CASE_REGEX.sub(_camel_case_replacer, text)


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
            loader=PackageLoader("patter_atleas_dynamic"),
            autoescape=select_autoescape(
                enabled_extensions=("html", "jinja2", "css"), default_for_string=True
            ),
        )
        self._jinja.filters["resource"] = self._resource
        self._jinja.filters["markdown"] = self._markdown
        self._jinja.filters["split_camel_case"] = split_camel_case
        self._mistune = create_markdown(
            escape=True, plugins=["table", "footnotes", "math", "url", "task_lists"]
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
            asset_url = f"/assets/{sha3_256(byte_content).hexdigest()}{''.join(suffixes)}"
            self._resource_map[url] = asset_url
            self._resource_bytes[asset_url] = byte_content
        except Exception:
            print(f"Failed to download resource: {url}")
            self._resource_map[url] = "/assets/empty.svg#" + url

        return self._resource_map[url]

    def _markdown(self, markdown: str):
        if not markdown:
            return "–"
        html = self._mistune(markdown.strip())
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

        asset_url = "/ui/assets/empty.svg"
        self._resource_map[""] = asset_url
        self._resource_bytes[asset_url] = svg_bytes

    def render_styles(self) -> str:
        template = self._jinja.get_template("styles.css")
        return template.render()

    def render_index(self, atlas: AtlasContent) -> str:
        template = self._jinja.get_template("languages.jinja2")
        return template.render(atlas=atlas, is_planqk=self.is_planqk)

    def render_language_overview(self, atlas: AtlasContent, language: PatternLanguage) -> str:
        template = self._jinja.get_template("language-overview.jinja2")
        return template.render(atlas=atlas, language=language, is_planqk=self.is_planqk)

    def render_pattern(self, atlas: AtlasContent, pattern: Pattern, language: PatternLanguage) -> str:
        template = self._jinja.get_template("pattern.jinja2")
        return template.render(atlas=atlas, pattern=pattern, language=language, is_planqk=self.is_planqk)

