# Copyright 2026 QHAna plugin runner contributors.
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

"""Shared test data and stubs for the router plugin tests."""

import json
from itertools import count
from typing import Dict, List, Optional

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from router.schemas import (
    AGGREGATOR_PLUGIN,
    MAPPING_PLUGIN,
    MDS_PLUGIN,
    PCA_PLUGIN,
    PIPELINE_PLUGINS,
    TRANSFORMERS_PLUGIN,
    VECTOR_CONCAT_PLUGIN,
    WU_PALMER_PLUGIN,
    InputParameters,
    InputParametersSchema,
)
from router.tasks import start_routing_task

from tests.utils import MockResponse

BASE_URL = "http://localhost:5005"
PLUGIN_BASE = f"{BASE_URL}/plugins"
FILE_BASE = f"{BASE_URL}/files"

ENTITIES_URL = f"{FILE_BASE}/1/entities.csv"
METADATA_URL = f"{FILE_BASE}/2/attribute_metadata.json"
TAXONOMIES_URL = f"{FILE_BASE}/3/taxonomies.zip"
WEBHOOK_URL = f"{PLUGIN_BASE}/router/1/webhook/"

PLUGIN_URLS = {key: f"{PLUGIN_BASE}/{name}/" for key, name in PIPELINE_PLUGINS.items()}


# --- INPUT PARAMETERS ---


def router_payload(**overrides) -> dict:
    """Form payload for ``InputParametersSchema`` with valid defaults."""
    payload = {
        "entitiesUrl": ENTITIES_URL,
        "entitiesMetadataUrl": METADATA_URL,
        "taxonomiesZipUrl": TAXONOMIES_URL,
        "includeIntermediateResultsInOutput": False,
        "rootIsPartOfHierarchy": False,
        "distanceMetric": "euclidean",
        "transformer": "linear_inverse",
        "mdsDimensions": 2,
        "metric": "metric_mds",
        "nInit": 4,
        "maxIter": 300,
        "missingDataHandling": "mean",
        "concatOutput": False,
        "outputFormat": "csv",
        "reduceDimensions": False,
        "pcaType": "normal",
        "pcaDimensions": 1,
        "solver": "auto",
        "tol": 0,
        "iteratedPower": 0,
    }
    payload.update(overrides)
    return payload


def router_params(**overrides) -> InputParameters:
    """Loaded ``InputParameters`` for helpers that take the parsed parameters."""
    return InputParametersSchema().load(router_payload(**overrides))


# --- PROCESSING TASK ---

DEFAULT_SELECTIONS = {"attr1": WU_PALMER_PLUGIN, "attr2": MAPPING_PLUGIN}


def make_router_task(
    *,
    selections: Optional[dict] = None,
    data: Optional[dict] = None,
    **payload_overrides,
) -> ProcessingTask:
    """Persist a ``ProcessingTask`` in the state the routing step leaves behind.

    ``data`` is applied last so a test can override any of the derived keys.
    """
    if selections is None:
        selections = dict(DEFAULT_SELECTIONS)

    db_task = ProcessingTask(
        task_name=start_routing_task.name,
        parameters=json.dumps(router_payload(**payload_overrides)),
    )
    db_task.data["webhook_url"] = WEBHOOK_URL
    db_task.data["plugin_urls"] = dict(PLUGIN_URLS)
    db_task.data["routing_selections"] = dict(selections)

    queue: List[str] = []
    for plugin in (WU_PALMER_PLUGIN, MAPPING_PLUGIN):
        attributes = [attr for attr, opt in selections.items() if opt == plugin]
        if attributes:
            db_task.data[f"{plugin}_attributes"] = "\n".join(attributes)
            queue.append(plugin)

    db_task.data["pipeline_queue"] = queue
    db_task.data["current_pipeline"] = queue[0] if queue else None
    db_task.progress_value = 1

    if data:
        db_task.data.update(data)

    db_task.save(commit=True)
    return db_task


# --- ENTITY / TAXONOMY INPUT FILES ---

# ``missing_tax`` references a taxonomy that is absent from the zip and
# ``composer`` references a plain file, so both must be skipped by the
# preprocessing task.
ENTITIES_CSV = (
    "ID,href,genre,instrumentation,composer,missing_tax\n"
    "e1,,g1,i1,c1,m1\n"
    "e2,,g2,i2,c2,m2\n"
)

ATTRIBUTE_METADATA = [
    {
        "ID": "genre",
        "type": "genre",
        "title": "Genre",
        "refTarget": "taxonomies.zip:t_genre.json",
    },
    {
        "ID": "instrumentation",
        "type": "instrumentation",
        "title": "Instrumentation",
        "refTarget": "taxonomies.zip:t_instrumentation.json",
    },
    {
        "ID": "composer",
        "type": "composer",
        "title": "Composer",
        "refTarget": "people.csv",
    },
    {
        "ID": "missing_tax",
        "type": "missing_tax",
        "title": "Missing taxonomy",
        "refTarget": "taxonomies.zip:t_missing.json",
    },
    {
        "ID": "not_in_entities",
        "type": "not_in_entities",
        "title": "Not in entities",
        "refTarget": "taxonomies.zip:t_genre.json",
    },
]

# ``mapping_raw`` values decide the pipeline recommendation per taxonomy.
TAXONOMY_MEMBERS = {
    "t_genre.json": json.dumps(
        {
            "entities": [
                {"ID": "g1", "mapping_raw": "1,2"},
                {"ID": "g2", "mapping_raw": ""},
            ]
        }
    ),
    "t_instrumentation.json": json.dumps(
        {"entities": [{"ID": "i1", "mapping_raw": ""}, {"ID": "i2", "mapping_raw": ""}]}
    ),
}


def input_file_responses() -> Dict[str, MockResponse]:
    """The three input files the router form points at."""
    return {
        ENTITIES_URL: MockResponse(ENTITIES_URL, "text/csv", text=ENTITIES_CSV),
        METADATA_URL: MockResponse(
            METADATA_URL, "application/json", json_data=ATTRIBUTE_METADATA
        ),
        TAXONOMIES_URL: MockResponse.from_zip(TAXONOMIES_URL, TAXONOMY_MEMBERS),
    }


def mock_open_url(monkeypatch, responses: Dict[str, MockResponse]):
    """Serve ``open_url`` from an in-memory url mapping in every router module."""

    def _open_url(url, *args, **kwargs):
        assert url in responses, f"Unexpected open_url for {url}"
        return responses[url]

    for module in ("router.tasks", "router.tasks_helpers", "router.tasks_pipeline_steps"):
        monkeypatch.setattr(f"{module}.open_url", _open_url)
    return _open_url


# --- SUB-PLUGIN HTTP STUB ---

VECTOR_CSV = "ID,dim0,dim1,dim2\nent1,1.0,2.0,3.0\nent2,4.0,5.0,6.0\n"

# Outputs each sub-plugin publishes when it finishes.
PLUGIN_OUTPUT_TYPES = {
    WU_PALMER_PLUGIN: ("relation/element-similarities",),
    MAPPING_PLUGIN: ("relation/element-distances",),
    TRANSFORMERS_PLUGIN: ("relation/element-distances",),
    AGGREGATOR_PLUGIN: ("relation/attribute-distances",),
    MDS_PLUGIN: ("entity/vector",),
    VECTOR_CONCAT_PLUGIN: ("entity/vector",),
    PCA_PLUGIN: ("entity/vector", "custom/pca-metadata", "custom/plot"),
}

_PLAIN_OUTPUTS = {
    "custom/pca-metadata": ("application/json", '{"explained_variance": [0.9]}'),
    "custom/plot": ("text/html", "<html>plot</html>"),
}

_SERVER_INSTANCES = count(1)


class PluginServer:
    """In-memory stand-in for the sub-plugin HTTP API.

    Records the payloads posted by ``run_pipeline_step``, hands out a task url
    per sub-plugin, and serves the result status and output files.
    """

    def __init__(self, *, subscribe_result: bool = True):
        self.subscribe_result = subscribe_result
        self.posts: List[tuple] = []
        self.subscriptions: List[dict] = []
        self.statuses: Dict[str, str] = {}
        self.files: Dict[str, MockResponse] = dict(input_file_responses())
        self._task_urls: Dict[str, str] = {}
        self._plugin_by_task_url: Dict[str, str] = {}
        self._counters: Dict[str, int] = {}
        self._indices: Dict[str, int] = {}
        self._outputs: Dict[str, List[dict]] = {}
        # The router locks a sub-task url once; a shared test database would
        # otherwise let one test's lock block the next one.
        self._instance = next(_SERVER_INSTANCES)

        for index, key in enumerate(PIPELINE_PLUGINS, start=1):
            self._indices[key] = index
            task_url = self._new_task_url(key)
            outputs = []
            for data_type in PLUGIN_OUTPUT_TYPES[key]:
                slug = data_type.replace("/", "_")
                href = f"{task_url}files/{slug}/download/"
                outputs.append({"dataType": data_type, "href": href, "name": slug})
                self.files[href] = self._output_response(href, key, data_type)
            self._outputs[key] = outputs

    def _new_task_url(self, plugin: str) -> str:
        """Hand out a fresh sub-task url, as a real plugin does per request."""
        self._counters[plugin] = self._counters.get(plugin, 0) + 1
        url = (
            f"{PLUGIN_BASE}/{PIPELINE_PLUGINS[plugin]}/tasks/"
            f"{self._instance}-{self._indices[plugin]}{self._counters[plugin]:02d}/"
        )
        self._task_urls[plugin] = url
        self._plugin_by_task_url[url] = plugin
        self.statuses[url] = "SUCCESS"
        return url

    @staticmethod
    def _output_response(href: str, plugin: str, data_type: str) -> MockResponse:
        if data_type in _PLAIN_OUTPUTS:
            content_type, text = _PLAIN_OUTPUTS[data_type]
            return MockResponse(href, content_type, text=text)
        if data_type == "entity/vector" and plugin in (VECTOR_CONCAT_PLUGIN, PCA_PLUGIN):
            return MockResponse(href, "text/csv", text=VECTOR_CSV)
        return MockResponse.from_zip(href, {"attr1.json": "{}"})

    # -- accessors -------------------------------------------------------

    def task_url(self, plugin: str) -> str:
        return self._task_urls[plugin]

    def outputs(self, plugin: str) -> List[dict]:
        return self._outputs[plugin]

    def output_url(self, plugin: str, data_type: str) -> str:
        for output in self._outputs[plugin]:
            if output["dataType"] == data_type:
                return output["href"]
        raise KeyError(f"{plugin} publishes no {data_type}")

    def payload(self, plugin: str) -> dict:
        """The payload of the first POST made to ``plugin``."""
        for posted_plugin, payload in self.posts:
            if posted_plugin == plugin:
                return payload
        raise AssertionError(f"No POST recorded for {plugin}")

    def post_count(self, plugin: str) -> int:
        return sum(1 for posted_plugin, _ in self.posts if posted_plugin == plugin)

    # -- request handlers ------------------------------------------------

    def _plugin_for_url(self, url: str) -> Optional[str]:
        for key, base in PLUGIN_URLS.items():
            if url.startswith(base):
                return key
        return None

    def _post(self, url, **kwargs):
        plugin = self._plugin_for_url(url)
        assert plugin is not None, f"Unexpected POST to {url}"
        self.posts.append((plugin, kwargs.get("data")))
        return MockResponse(
            url,
            "text/html",
            status_code=303,
            headers={"Location": self._new_task_url(plugin)},
        )

    def _get(self, url, **kwargs):
        if url in self.statuses:
            plugin = self._plugin_by_task_url[url]
            status = self.statuses[url]
            return MockResponse(
                url,
                "application/json",
                json_data={
                    "status": status,
                    "outputs": self._outputs[plugin] if status == "SUCCESS" else [],
                    "links": [{"type": "subscribe", "href": url + "subscribe/"}],
                },
            )
        if url in self.files:
            return self.files[url]
        raise AssertionError(f"Unexpected GET to {url}")

    def _subscribe(self, **kwargs):
        self.subscriptions.append(kwargs)
        return self.subscribe_result

    def install(self, monkeypatch):
        monkeypatch.setattr(
            "router.tasks_helpers.get_plugin_endpoint", lambda url: url + "process/"
        )
        monkeypatch.setattr("router.tasks_helpers.subscribe", self._subscribe)
        monkeypatch.setattr("requests.post", self._post)
        monkeypatch.setattr("requests.get", self._get)
        mock_open_url(monkeypatch, self.files)
        return self


def capture_pipeline_step(monkeypatch) -> List[dict]:
    """Replace ``run_pipeline_step`` and record the kwargs of every call."""
    calls: List[dict] = []
    monkeypatch.setattr(
        "router.tasks_pipeline_steps.run_pipeline_step",
        lambda **kwargs: calls.append(kwargs),
    )
    return calls
