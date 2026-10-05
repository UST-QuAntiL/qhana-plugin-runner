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

"""Tests for the PCA settings of the routing step micro frontend."""

import json
from http import HTTPStatus

import pytest
from flask import url_for

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from feature_engineering_pipeline import FEATURE_ENGINEERING_PIPELINE_BLP
from feature_engineering_pipeline.schemas import WU_PALMER_PLUGIN
from feature_engineering_pipeline.tests.data import make_router_task

TAXONOMY_ATTRIBUTES = ["genre", "instrumentation"]


@pytest.fixture
def db_task(request) -> ProcessingTask:
    """Task in the state the first step leaves behind, with two attributes."""
    overrides = getattr(request, "param", {})
    return make_router_task(
        selections={},
        data={"taxonomy_attributes": list(TAXONOMY_ATTRIBUTES)},
        **overrides,
    )


def _ui_url(db_task: ProcessingTask) -> str:
    return url_for(
        f"{FEATURE_ENGINEERING_PIPELINE_BLP.name}.RoutingStepFrontend", db_id=db_task.id
    )


def _process_url(db_task: ProcessingTask) -> str:
    return url_for(
        f"{FEATURE_ENGINEERING_PIPELINE_BLP.name}.RoutingStepView", db_id=db_task.id
    )


def _parameters(db_task: ProcessingTask) -> dict:
    DB.session.expire_all()
    return json.loads(ProcessingTask.get_by_id(db_task.id).parameters)


@pytest.mark.parametrize("db_task", [{"concatOutput": True}], indirect=True)
def test_routing_step_renders_the_pca_settings(client, db_task):
    resp = client.get(_ui_url(db_task))

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)
    assert 'id="reduce_dimensions"' in body
    assert 'id="pca_dimensions"' in body
    assert 'id="estimated-dimensions"' in body


@pytest.mark.parametrize(
    "db_task", [{"concatOutput": True, "mdsDimensions": 3}], indirect=True
)
def test_routing_step_estimate_uses_the_mds_dimensions(client, db_task):
    resp = client.get(_ui_url(db_task))

    assert "const mdsDimensions = 3;" in resp.get_data(as_text=True)


def test_routing_step_omits_the_pca_settings_without_concatenation(client, db_task):
    resp = client.get(_ui_url(db_task))

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)
    assert 'id="pca_dimensions"' not in body
    assert 'Enable "Concat output" in the' in body


@pytest.mark.parametrize("db_task", [{"concatOutput": True}], indirect=True)
def test_routing_step_stores_the_pca_settings(client, db_task):
    resp = client.post(
        _process_url(db_task),
        data={
            "pipeline_genre": WU_PALMER_PLUGIN,
            "reduceDimensions": "on",
            "pcaType": "kernel",
            "pcaDimensions": "4",
            "solver": "full",
            "tol": "0.5",
            "iteratedPower": "7",
        },
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    parameters = _parameters(db_task)
    assert parameters["reduceDimensions"] is True
    assert parameters["pcaType"] == "kernel"
    assert parameters["pcaDimensions"] == 4
    assert parameters["solver"] == "full"
    assert parameters["tol"] == 0.5
    assert parameters["iteratedPower"] == 7
    # the selections stay in ``data``, only the PCA settings are merged
    assert ProcessingTask.get_by_id(db_task.id).data["routing_selections"] == {
        "genre": WU_PALMER_PLUGIN
    }


def test_routing_step_ignores_pca_settings_without_concatenation(client, db_task):
    """The form omits the section, so a submitted value is not honored."""
    resp = client.post(
        _process_url(db_task),
        data={"pipeline_genre": WU_PALMER_PLUGIN, "reduceDimensions": "on"},
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert _parameters(db_task)["reduceDimensions"] is False
