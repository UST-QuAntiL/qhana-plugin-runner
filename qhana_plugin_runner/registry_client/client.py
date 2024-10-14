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

from http import HTTPStatus
from logging import Logger
from time import time
from typing import Any, Dict, Literal, Optional, Sequence, Union

from flask import Flask
from flask.globals import current_app
from requests import request

from .types import ApiLink, ApiResponse, match_api_link
from ..util.logging import get_logger

_API_RESPONSE_MIN_KEYS = {"data", "links"}

_REGISTRY_CLIENT_LOGGER = "registry_client"


class PluginRegistryClient:
    """A minimal client for the qhana plugin registry."""

    def __init__(self, app: Optional[Flask] = None) -> None:
        self._cache: Dict[str, ApiResponse] = {}
        self._last_cache_clear = time()
        self._plugin_registry_url: Optional[str] = None
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        self._plugin_registry_url = app.config.get("PLUGIN_REGISTRY_URL", None)

    @property
    def ready(self) -> bool:
        return bool(self.plugin_registry_url)

    @property
    def plugin_registry_url(self) -> Optional[str]:
        return current_app.config.get("PLUGIN_REGISTRY_URL", self._plugin_registry_url)

    def _cache_response(self, response: ApiResponse):
        """Cache a response if it is an "api" response.

        Args:
            response (ApiResponse): the response to cache
        """
        self_link = response.data["self"]
        if match_api_link(self_link, {"api"}):
            cachable_response = ApiResponse(
                data=response.data, links=response.links, keyed_links=response.keyed_links
            )
            self._cache[self_link["href"]] = cachable_response

    def _fetch_by_url(
        self,
        url: str,
        query_params: Optional[dict] = None,
        body: Optional[bytes] = None,
        json: Optional[Any] = None,
        method: Literal["get", "put", "post", "delete"] = "get",
    ) -> ApiResponse:
        """Fetch a response by url.

        This method is not intended to be called directly!

        Args:
            url (str): the url to fetch
            query_params (Optional[dict], optional): additional query params to use. Defaults to None.
            body (Optional[bytes]): a body to use for put or post requests. Defaults to None.
            json (Optional[Any]): a json serializable object to use as a body for put or post requests. Defaults to None.
            method (Literal["get", "put", "post", "delete"]): the http method to use for the request. Defaults to "get".

        Raises:
            ValueError: if the response does not appear to be an ApiResponse

        Returns:
            ApiResponse: the response
        """
        if (time() - self._last_cache_clear) > (24 * 60 * 60):
            self._cache = {}

        if not query_params:
            cached = self._cache.get(url, None)
            if cached is not None:
                return cached

        if current_app:
            get_logger(current_app, _REGISTRY_CLIENT_LOGGER).debug(
                f"Requesting URL '{url}' with query params {query_params}"
            )
        response = request(method, url, params=query_params, data=body, json=json)

        if response.status_code == HTTPStatus.NO_CONTENT:
            return ApiResponse(
                data={"self": {"href": url, "rel": [], "resourceType": "accepted"}},
                links=[],
            )

        response_data = response.json()
        if response_data.keys() < _API_RESPONSE_MIN_KEYS:
            raise ValueError(
                "ApiResponse must contain the links and data!", response_data
            )

        embedded_data = {
            e["data"]["self"]["href"]: ApiResponse(
                data=e["data"],
                links=e["links"],
                keyed_links=e.get("keyedLinks", tuple()),
            )
            for e in response_data.get("embedded", tuple())
        }

        api_response = ApiResponse(
            data=response_data["data"],
            links=response_data["links"],
            keyed_links=response_data.get("keyedLinks", tuple()),
            embedded=embedded_data if embedded_data else None,
        )

        self._cache_response(api_response)

        return api_response

    def fetch_by_api_link(
        self,
        link: ApiLink,
        query_params: Optional[dict] = None,
        body: Optional[bytes] = None,
        json: Optional[Any] = None,
    ) -> ApiResponse:
        """Fetch a response for a specific api link.

        Args:
            link (ApiLink): the link to the resource
            query_params (Optional[dict], optional): additional query params to use. Defaults to None.
            body (Optional[bytes]): a body to use for put or post requests. Defaults to None.
            json (Optional[Any]): a json serializable object to use as a body for put or post requests. Defaults to None.

        Returns:
            ApiResponse: the api response
        """
        rels = link.get("rel", ())
        method = "get"
        if "post" in rels:
            method = "post"
        elif "put" in rels:
            method = "put"
        elif "delete" in rels:
            method = "delete"
        return self._fetch_by_url(
            link["href"], query_params=query_params, body=body, json=json, method=method
        )

    def search_by_rel(
        self,
        rel: Union[str, Sequence[str]],
        query_params: Optional[dict] = None,
        body: Optional[bytes] = None,
        json: Optional[Any] = None,
        allow_collection_resource: bool = True,
        base: Optional[ApiResponse] = None,
        ignore_base_match: bool = False,
    ) -> Optional[ApiResponse]:
        """search the api for a resource matching the given rel starting from the given base resource.

        This method is intended to be used inside a requests session.

        Args:
            rel (Union[str, Sequence[str]]): the rel (or rels) to search for
            query_params (Optional[dict], optional): additional query params to use (will be used for every request during the search). Defaults to None.
            body (Optional[bytes]): a body to use for put or post requests. Defaults to None.
            json (Optional[Any]): a json serializable object to use as a body for put or post requests. Defaults to None.
            allow_collection_resource (bool, optional): if False this method will not return a collection resource as its final result. Defaults to True.
            base (Optional[ApiResponse], optional): the starting point of the search (in form of a resource). Defaults to None.
            ignore_base_match (bool, optional): if True ignore a match of the given base resource (use this if you pass a base and do not want it back immediately). Defaults to False.

        Raises:
            ValueError: if no base is given and the plugin registry url is None

        Returns:
            Optional[ApiResponse]: the found api response
        """
        if base is None:
            if self.plugin_registry_url is None:
                raise ValueError(
                    "The plugin registry url must be set if no base is provided!"
                )
            base = self._fetch_by_url(self.plugin_registry_url)

        if not ignore_base_match and base.matches_rel(rel):
            if allow_collection_resource or not base.is_collection_resource():
                return base

        direct_matches = base.get_links_by_rel(rel)
        if base.is_collection_resource():
            direct_matches = base.data.get("items", direct_matches)

        # fetch potential results
        for link in direct_matches:
            result = base.embedded.get(link["href"], None) if base.embedded else None
            if result is not None and result.matches_rel(rel):
                if allow_collection_resource or not result.is_collection_resource():
                    return result
                final_result = self.search_by_rel(
                    rel,
                    query_params,
                    body=body,
                    json=json,
                    allow_collection_resource=allow_collection_resource,
                    base=result,
                )
                if final_result is not None:
                    return final_result
            result = self.fetch_by_api_link(
                link, query_params=query_params, body=body, json=json
            )
            if result.matches_rel(rel):
                if allow_collection_resource or not result.is_collection_resource():
                    return result
                final_result = self.search_by_rel(
                    rel,
                    query_params,
                    body=body,
                    json=json,
                    allow_collection_resource=allow_collection_resource,
                    base=result,
                )
                if final_result is not None:
                    return final_result

        # expand search to include api links
        for api_link in base.get_links_by_rel("api"):
            new_base = self.fetch_by_api_link(api_link, query_params=query_params)
            final_result = self.search_by_rel(
                rel,
                query_params,
                allow_collection_resource=allow_collection_resource,
                base=new_base,
            )
            if final_result is not None:
                return final_result

        return None

    def fetch_by_rel(
        self,
        rels: Sequence[Union[str, Sequence[str]]],
        query_params: Optional[dict] = None,
        body: Optional[bytes] = None,
        json: Optional[Any] = None,
        allow_collection_resource: bool = True,
        base: Optional[ApiResponse] = None,
    ) -> Optional[ApiResponse]:
        """Search the api following the provided path of rels.

        This method is intended to be used inside a requests session.

        Each step in the rels path will be searched for by ``search_by_rel``.

        Args:
            rels (Sequence[Union[str, Sequence[str]]]): the path of rels to follow
            query_params (Optional[dict], optional): additional query params to use (will be used for every request during the search). Defaults to None.
            body (Optional[bytes]): a body to use for put or post requests. Defaults to None.
            json (Optional[Any]): a json serializable object to use as a body for put or post requests. Defaults to None.
            allow_collection_resource (bool, optional): if False this method will not return a collection resource as its final result. Defaults to True.
            base (Optional[ApiResponse], optional): a base resource to start the search from. Defaults to None.

        Returns:
            Optional[ApiResponse]: the found api response
        """
        response = base
        for index, rel in enumerate(rels):
            is_last = index == len(rels) - 1
            if is_last:
                if rel in ("put", "post", "delete") or not set(rel).isdisjoint(
                    {"put", "post", "delete"}
                ):
                    for link in response.links:
                        if match_api_link(
                            link, rel=set([rel]) if isinstance(rel, str) else set(rel)
                        ):
                            self.fetch_by_api_link(
                                link,
                                query_params=query_params,
                                body=body,
                                json=json,
                            )
            response = self.search_by_rel(
                rel=rel,
                query_params=query_params,
                body=body,
                json=json,
                base=response,
                allow_collection_resource=(not is_last or allow_collection_resource),
                ignore_base_match=True,
            )
            if response is None:
                return None

        if response is not None and response.matches_rel(rels[-1]):
            return response

        return None
