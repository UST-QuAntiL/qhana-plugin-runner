# Copyright 2023 QHAna plugin runner contributors.
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
import os
from dataclasses import dataclass
from typing import Iterator, List

from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate

from plugins.rest_connector.openapi import (
    parse_spec,
    get_endpoint_paths,
    get_endpoint_methods,
    get_endpoint_method_summary,
)


@dataclass
class Endpoint:
    path: str
    method: str
    summary: str


def get_endpoints(spec_url: str) -> Iterator[Endpoint]:
    spec = parse_spec(spec_url)
    endpoints = get_endpoint_paths(spec)

    for endpoint in endpoints:
        methods = get_endpoint_methods(spec, endpoint)

        for method in methods:
            summary = get_endpoint_method_summary(spec, endpoint, method)

            yield Endpoint(endpoint, method, summary)


def wrapper(chat_model: BaseChatModel, user_request: str):
    def check_if_endpoint_is_relevant(endpoint: Endpoint) -> bool:
        prompt_template = ChatPromptTemplate.from_template(
            'The user wants to do the following: "{user_request}"\nCan the following REST endpoint be used to '
            "accomplish that? Only answer with yes or no.\nEndpoint: {path}\nMethod: {method}\nSummary: {summary}"
        )
        response = chat_model(
            prompt_template.format_prompt(
                user_request=user_request,
                path=endpoint.path,
                method=endpoint.method,
                summary=endpoint.summary,
            ).to_messages()
        )

        return "yes" in response.content.lower()

    return check_if_endpoint_is_relevant


# TODO: use vector store to filter endpoints, before running an LLM, configure with flags (e.g. env vars)
# TODO: figure out which HTTP methods are relevant for a user request
# TODO: performance counter, store in extra


def get_relevant_endpoints(
    chat: BaseChatModel, spec_url: str, user_request: str
) -> List[Endpoint]:
    endpoints = get_endpoints(spec_url)
    relevant_endpoints = filter(
        wrapper(chat, user_request),
        endpoints,
    )

    return list(relevant_endpoints)


def _example():
    chat = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])
    endpoints = get_endpoints(
        "https://raw.githubusercontent.com/swagger-api/swagger-petstore/master/src/main/resources/openapi.yaml"
    )
    user_request = "I want to add a new pet."
    relevant_endpoints = filter(
        wrapper(chat, user_request),
        endpoints,
    )

    for end in relevant_endpoints:
        print(end)


if __name__ == "__main__":
    _example()
