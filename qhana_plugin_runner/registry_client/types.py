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

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set, TypedDict, Union


class ApiLinkBase(TypedDict):
    href: str
    resourceType: str
    rel: Sequence[str]


class ApiLink(ApiLinkBase, total=False):
    schema: str
    doc: str
    name: str
    resourceKey: Dict[str, str]


class KeyedApiLink(ApiLink, total=False):
    key: Sequence[str]
    queryKey: Sequence[str]


class BaseApiObject(TypedDict):
    self: ApiLink


class ApiObject(BaseApiObject, total=False):
    items: Sequence[ApiLink]


def match_api_link(link: ApiLink, rel: Set[str], resource_type: Optional[str] = None):
    if resource_type is not None and link["resourceType"] != resource_type:
        return False
    link_rels = set(link["rel"])
    link_rels.add(link["resourceType"])
    return link_rels >= rel


@dataclass
class ApiResponse:
    data: ApiObject
    links: Sequence[ApiLink]
    keyed_links: Sequence[KeyedApiLink] = tuple()
    embedded: "Optional[Dict[str, ApiResponse]]" = None

    @property
    def self(self) -> str:
        return self.data["self"]["href"]

    def get_links_by_rel(
        self, rel: Union[str, Sequence[str]], resource_type: Optional[str] = None
    ):
        search_rels: Set[str] = set()
        if isinstance(rel, str):
            search_rels.add(rel)
        else:
            search_rels.update(rel)
        if resource_type is not None:
            search_rels.add(resource_type)
        return tuple(
            link
            for link in self.links
            if match_api_link(link, search_rels, resource_type=resource_type)
        )

    def matches_rel(
        self, rel: Union[str, Sequence[str]], resource_type: Optional[str] = None
    ) -> bool:
        link = self.data["self"]
        if resource_type is not None and link["resourceType"] != resource_type:
            return False
        if isinstance(rel, str):
            return link["resourceType"] == rel or rel in link["rel"]
        else:
            return all(link["resourceType"] == r or r in link["rel"] for r in rel)

    def is_collection_resource(self) -> bool:
        return self.matches_rel("collection") or self.matches_rel("page")
