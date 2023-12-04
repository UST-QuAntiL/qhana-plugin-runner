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

from celery.utils.log import get_task_logger
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

from plugins.rest_connector.openapi import (
    parse_spec,
    get_endpoint_paths,
    get_endpoint_methods,
    get_endpoint_method_summary,
)


# TODO: add env vars to readme
def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    openai_api_key: str | None = None,
) -> BaseChatModel:
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "openai")

    if model is None:
        model = os.environ.get("LLM_MODEL", "chatgpt-3.5")

    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY", None)

    if provider == "openai":
        if openai_api_key is None:
            raise ValueError(f"OpenAI API key not provided.")

        return ChatOpenAI(model=model, openai_api_key=openai_api_key)
    elif provider == "ollama":
        return ChatOllama(model=model)
    else:
        raise ValueError(f"Unknown provider {provider}")


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


def get_relevant_endpoints_from_llm(
    chat: BaseChatModel, endpoints: List[Endpoint], user_request: str
) -> List[Endpoint]:
    relevant_endpoints = filter(
        wrapper(chat, user_request),
        endpoints,
    )

    return list(relevant_endpoints)


def get_relevant_endpoints_from_vector_store(
    spec_url: str, user_request: str, max_relevant_endpoints: int = 5
) -> List[Endpoint]:
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docs: List[Document] = []
    endpoints = get_endpoints(spec_url)

    for endpoint in endpoints:
        docs.append(
            Document(
                page_content=endpoint.summary,
                metadata={
                    "path": endpoint.path,
                    "method": endpoint.method,
                    "summary": endpoint.summary,
                },
            )
        )

    db = Chroma.from_documents(docs, embedding_function)
    relevant_docs = db.similarity_search(user_request, k=max_relevant_endpoints)

    return [
        Endpoint(
            path=doc.metadata["path"],
            method=doc.metadata["method"],
            summary=doc.metadata["summary"],
        )
        for doc in relevant_docs
    ]


def _example_iterator():
    chat = get_chat_model(provider="ollama", model="neural-chat:7b-v3.1")
    endpoints = get_endpoints(
        "https://raw.githubusercontent.com/swagger-api/swagger-petstore/master/src/main/resources/openapi.yaml"
    )
    user_request = "I want to update the data of a pet."
    relevant_endpoints = filter(
        wrapper(chat, user_request),
        endpoints,
    )

    for end in relevant_endpoints:
        print(end)


def _example_list():
    chat = get_chat_model(provider="ollama", model="neural-chat:7b-v3.1")
    endpoints = get_endpoints(
        "https://raw.githubusercontent.com/swagger-api/swagger-petstore/master/src/main/resources/openapi.yaml"
    )
    user_request = "I want to update the data of a pet."
    relevant_endpoints = get_relevant_endpoints_from_llm(
        chat, list(endpoints), user_request
    )

    for end in relevant_endpoints:
        print(end)


def _example_llm_with_vector_store():
    chat = get_chat_model(provider="ollama", model="neural-chat:7b-v3.1")
    user_request = "I want to update the data of a pet."

    prefiltered_endpoints = get_relevant_endpoints_from_vector_store(
        "https://raw.githubusercontent.com/swagger-api/swagger-petstore/master/src/main/resources/openapi.yaml",
        "I want to add a new pet.",
        3,
    )

    print("with vector store prefiltered endpoints:")

    for end in prefiltered_endpoints:
        print(end)

    print()

    relevant_endpoints = get_relevant_endpoints_from_llm(
        chat, list(prefiltered_endpoints), user_request
    )

    print("with LLM filtered endpoints:")

    for end in relevant_endpoints:
        print(end)


if __name__ == "__main__":
    _example_llm_with_vector_store()
