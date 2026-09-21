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

import re
from http import HTTPStatus
from urllib.parse import urlsplit

import pytest
from flask import current_app, url_for

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from feature_engineering_pipeline import ROUTER_BLP, Router
from feature_engineering_pipeline.routes import INPUT_FIELD_GROUPS
from feature_engineering_pipeline.schemas import (
    INCLUDE_NUMERIC,
    MAPPING_PLUGIN,
    NONE_PLUGIN,
    PIPELINE_OPTIONS,
    WU_PALMER_PLUGIN,
    InputParametersSchema,
)
from feature_engineering_pipeline.tests.data import router_payload

from tests.utils import mock_task_dispatch


def _path(endpoint: str, **kwargs) -> str:
    return urlsplit(url_for(f"{ROUTER_BLP.name}.{endpoint}", **kwargs)).path


def _form(**overrides) -> dict:
    """The input parameters as a browser would submit them."""
    payload = router_payload(**overrides)
    return {
        key: ("true" if value else "false") if isinstance(value, bool) else value
        for key, value in payload.items()
    }


def test_metadata_endpoint_returns_full_descriptor(client):
    resp = client.get(url_for(f"{ROUTER_BLP.name}.PluginsView"))
    plugin = Router.instance

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_json()

    assert body["name"] == plugin.name
    assert body["type"] == "processing"
    assert len(body["entryPoint"]["dataOutput"]) == 8


def test_microfrontend_renders_form_fields(client):
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))
    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)

    assert "Entities URL" in body
    assert "Distance Metric" in body
    assert "Transformer" in body
    assert "Metric" in body
    assert "Concat output" in body
    assert "Output Format" in body


def test_process_valid_payload_redirects_to_task(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    valid_payload = {
        "entitiesUrl": "http://example.com/data.csv",
        "entitiesMetadataUrl": "http://example.com/meta.json",
        "taxonomiesZipUrl": "http://example.com/tax.zip",
        "distanceMetric": "euclidean",
        "transformer": "linear_inverse",
        "mdsDimensions": 2,
        "metric": "metric_mds",
        "nInit": 4,
        "maxIter": 300,
        "missingDataHandling": "mean",
        "reduceDimensions": False,
        "pcaType": "normal",
        "pcaDimensions": 1,
        "solver": "auto",
        "tol": 0,
        "iteratedPower": 0,
    }

    resp = client.post(url_for(f"{ROUTER_BLP.name}.ProcessView"), data=valid_payload)

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert re.fullmatch(r"/tasks/\d+/", urlsplit(resp.headers["Location"]).path)


def test_routing_step_frontend_renders_attribute_dropdowns(client):
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["instrumentation", "genre"]
    db_task.data["recommendations"] = {"genre": PIPELINE_OPTIONS[MAPPING_PLUGIN]}
    db_task.save(commit=True)

    resp = client.get(_path("RoutingStepFrontend", db_id=db_task.id))

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)
    assert 'name="pipeline_instrumentation"' in body
    assert 'name="pipeline_genre"' in body
    assert PIPELINE_OPTIONS[WU_PALMER_PLUGIN] in body
    assert PIPELINE_OPTIONS[MAPPING_PLUGIN] in body


def test_routing_step_process_records_selection_and_redirects(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = ProcessingTask(
        task_name="router_test",
        parameters='{"entities_url": "http://example.com/data.csv"}',
    )
    db_task.data["taxonomy_attributes"] = ["instrumentation", "genre"]
    db_task.save(commit=True)

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={
            "pipeline_instrumentation": WU_PALMER_PLUGIN,
            "pipeline_genre": MAPPING_PLUGIN,
        },
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert re.fullmatch(r"/tasks/\d+/", urlsplit(resp.headers["Location"]).path)

    db_task = ProcessingTask.get_by_id(db_task.id)

    assert db_task.data.get("routing_selections") == {
        "instrumentation": WU_PALMER_PLUGIN,
        "genre": MAPPING_PLUGIN,
    }


def test_webhook_view_accepts_status_events(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    # Create a dummy task in DB
    db_task = ProcessingTask(task_name="router_test")
    db_task.save(commit=True)

    resp = client.post(
        url_for(f"{ROUTER_BLP.name}.WebhookView", db_id=db_task.id),
        query_string={"source": "http://localhost/tasks/1/", "event": "status"},
    )

    assert resp.status_code == HTTPStatus.OK
    assert resp.get_data(as_text=True) == '"Webhook received"\n'


def test_metadata_data_inputs_match_the_input_schema(client):
    """A data input the UI cannot map to a form field would break the workflow."""
    resp = client.get(url_for(f"{ROUTER_BLP.name}.PluginsView"))
    data_input = resp.get_json()["entryPoint"]["dataInput"]
    form_keys = {field.data_key for field in InputParametersSchema().fields.values()}

    assert len(data_input) == 3
    assert {entry["parameter"] for entry in data_input} <= form_keys


def test_microfrontend_renders_every_field_group(client):
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))
    body = resp.get_data(as_text=True)

    for _key, title, _field_names in INPUT_FIELD_GROUPS:
        assert title in body


def test_microfrontend_prefills_the_defaults(client):
    """The MDS and PCA parameters are prefilled so the form can be submitted as is."""
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))
    body = resp.get_data(as_text=True)

    assert 'value="300"' in body  # maxIter
    assert 'value="4"' in body  # nInit


def test_microfrontend_reports_pca_without_concatenation(client):
    """The cross field rule has to reach the user before the task is started."""
    resp = client.post(
        url_for(f"{ROUTER_BLP.name}.MicroFrontend"),
        data=_form(concatOutput=False, reduceDimensions=True),
    )

    assert resp.status_code == HTTPStatus.OK
    assert "has to be enabled as well" in resp.get_data(as_text=True)


def test_microfrontend_accepts_an_empty_form(client):
    """The frontend is validated partially, an untouched form is not an error."""
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))

    assert resp.status_code == HTTPStatus.OK
    assert "qhana-error-message" not in resp.get_data(as_text=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"entitiesUrl": "not-a-url"},
        {"mdsDimensions": 0},
        {"transformer": "no_such_transformer"},
        {"outputFormat": "xml"},
        {"concatOutput": False, "reduceDimensions": True},
    ],
)
def test_process_rejects_invalid_input(client, monkeypatch, overrides):
    mock_task_dispatch(monkeypatch)

    resp = client.post(url_for(f"{ROUTER_BLP.name}.ProcessView"), data=_form(**overrides))

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_routing_step_frontend_shows_the_inputs_of_the_first_step(client):
    db_task = ProcessingTask(
        task_name="router_test",
        parameters='{"entitiesUrl": "http://example.com/step-one.csv"}',
    )
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.save(commit=True)

    body = client.get(_path("RoutingStepFrontend", db_id=db_task.id)).get_data(
        as_text=True
    )

    assert "http://example.com/step-one.csv" in body


def _settings_task(**payload_overrides) -> ProcessingTask:
    """A task waiting in the routing step, with the parameters of the first step.

    ``hc__grundton`` is an attribute name that contains the separator between the
    attribute and its settings.
    """
    schema = InputParametersSchema()
    db_task = ProcessingTask(
        task_name="router_test",
        parameters=schema.dumps(schema.load(router_payload(**payload_overrides))),
    )
    db_task.data["taxonomy_attributes"] = ["hc__grundton", "genre"]
    db_task.save(commit=True)
    return db_task


def _tag(body: str, field_id: str) -> str:
    """The markup of the input or select element with the given id."""
    return body[body.index(f'id="{field_id}"') :].split(">")[0]


def test_routing_step_frontend_renders_the_pipeline_settings_per_attribute(client):
    """Each attribute gets its own copy of the settings of the first step, prefilled."""
    db_task = _settings_task(distanceMetric="cosine")

    body = client.get(_path("RoutingStepFrontend", db_id=db_task.id)).get_data(
        as_text=True
    )

    for attr in ("hc__grundton", "genre"):
        assert f'name="pipeline_{attr}__distanceMetric"' in body
    # the settings are submitted per attribute, not once for the whole run
    assert 'name="distanceMetric"' not in body
    assert body.count('value="cosine" selected') == 2


def test_routing_step_frontend_keeps_the_submitted_settings(client):
    """A second render shows what was submitted, not the settings of the first step."""
    db_task = _settings_task(distanceMetric="cosine", rootIsPartOfHierarchy=True)

    body = client.post(
        _path("RoutingStepFrontend", db_id=db_task.id),
        data={
            "pipeline_genre": WU_PALMER_PLUGIN,
            "pipeline_genre__distanceMetric": "manhatten",
        },
    ).get_data(as_text=True)

    genre = body[body.index('id="pipeline_genre__distance_metric"') :]
    assert 'value="manhatten" selected' in genre.split("</select>")[0]
    # the checkbox of the submitted block was unchecked, the untouched attribute
    # keeps the value of the first step
    assert "checked" not in _tag(body, "pipeline_genre__root_is_part_of_hierarchy")
    assert "checked" in _tag(body, "pipeline_hc__grundton__root_is_part_of_hierarchy")


def test_routing_step_stores_the_settings_of_every_attribute(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = _settings_task(rootIsPartOfHierarchy=True)

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={
            "pipeline_genre": MAPPING_PLUGIN,
            "pipeline_genre__distanceMetric": "cosine",
            "pipeline_genre__mdsDimensions": "3",
            "pipeline_hc__grundton": WU_PALMER_PLUGIN,
            "pipeline_hc__grundton__rootIsPartOfHierarchy": "true",
        },
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    data = ProcessingTask.get_by_id(db_task.id).data
    assert data["routing_selections"] == {
        "genre": MAPPING_PLUGIN,
        "hc__grundton": WU_PALMER_PLUGIN,
    }
    # an unchecked checkbox is not submitted at all, so it reads as False
    assert data["attribute_settings"] == {
        "genre": {
            "rootIsPartOfHierarchy": False,
            "distanceMetric": "cosine",
            "mdsDimensions": 3,
        },
        "hc__grundton": {"rootIsPartOfHierarchy": True},
    }


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("mdsDimensions", "0", "Must be greater than or equal to 1."),
        # a name that does not end in a known setting is read as an attribute name
        (
            "unknownSetting",
            "1",
            f"'1' is not one of {[*PIPELINE_OPTIONS, INCLUDE_NUMERIC]}.",
        ),
    ],
)
def test_routing_step_rejects_invalid_settings(
    client, monkeypatch, field, value, message
):
    mock_task_dispatch(monkeypatch)
    db_task = _settings_task()

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={"pipeline_genre": WU_PALMER_PLUGIN, f"pipeline_genre__{field}": value},
    )

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert resp.get_json()["errors"]["form"][f"pipeline_genre__{field}"] == [message]
    assert "attribute_settings" not in ProcessingTask.get_by_id(db_task.id).data


def test_routing_step_frontend_without_attributes(client):
    """Preprocessing may not find any taxonomy attribute at all."""
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.save(commit=True)

    resp = client.get(_path("RoutingStepFrontend", db_id=db_task.id))

    assert resp.status_code == HTTPStatus.OK
    assert 'name="pipeline_' not in resp.get_data(as_text=True)


def test_routing_step_frontend_rejects_an_unknown_task(client):
    with pytest.raises(KeyError):
        client.get(_path("RoutingStepFrontend", db_id=999999))


def test_routing_step_clears_the_previous_step(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.save(commit=True)
    db_task.add_next_step(
        href="http://localhost/step/",
        ui_href="http://localhost/step-ui/",
        step_id="routing-step",
    )
    db_task.save(commit=True)

    client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={"pipeline_genre": WU_PALMER_PLUGIN},
    )

    assert not ProcessingTask.get_by_id(db_task.id).has_uncleared_step


def test_routing_step_rejects_a_routing_without_any_pipeline(client, monkeypatch):
    """Routing every attribute to ``None`` would leave the pipeline queue empty."""
    mock_task_dispatch(monkeypatch)
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.save(commit=True)

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id), data={"pipeline_genre": NONE_PLUGIN}
    )

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_routing_step_rejects_an_unknown_pipeline(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.save(commit=True)

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id), data={"pipeline_genre": "quantum"}
    )

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_routing_step_rejects_an_unknown_task(client, monkeypatch):
    mock_task_dispatch(monkeypatch)

    with pytest.raises(KeyError):
        client.post(
            _path("RoutingStepView", db_id=999999),
            data={"pipeline_genre": WU_PALMER_PLUGIN},
        )


@pytest.fixture
def webhook_dispatch(monkeypatch):
    dispatched = []
    monkeypatch.setattr(
        "feature_engineering_pipeline.routes.handle_webhook_task.apply_async",
        lambda **kwargs: dispatched.append(kwargs),
    )
    return dispatched


def test_webhook_forwards_the_delivery_method(client, webhook_dispatch):
    db_task = ProcessingTask(task_name="router_test")
    db_task.save(commit=True)

    client.post(
        _path("WebhookView", db_id=db_task.id),
        query_string={
            "source": "http://localhost/tasks/1/",
            "event": "status",
            "via": "watchdog",
        },
    )

    assert webhook_dispatch == [
        {
            "kwargs": {
                "db_id": db_task.id,
                "source_url": "http://localhost/tasks/1/",
                "via": "watchdog",
            },
            "countdown": 2,
        }
    ]


@pytest.mark.parametrize(
    "query_string",
    [
        {"event": "status"},
        {"source": "http://localhost/tasks/1/"},
        {"source": "http://localhost/tasks/1/", "event": "steps"},
    ],
)
def test_webhook_ignores_events_it_cannot_act_on(client, webhook_dispatch, query_string):
    db_task = ProcessingTask(task_name="router_test")
    db_task.save(commit=True)

    resp = client.post(_path("WebhookView", db_id=db_task.id), query_string=query_string)

    assert resp.status_code == HTTPStatus.OK
    assert webhook_dispatch == []


def test_pipeline_options_are_offered_for_every_attribute(client):
    """Every option of the dropdown has to be accepted by the routing step."""
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.save(commit=True)

    body = client.get(_path("RoutingStepFrontend", db_id=db_task.id)).get_data(
        as_text=True
    )

    for key, label in PIPELINE_OPTIONS.items():
        assert f'value="{key}"' in body
        assert label in body


# --- NUMERIC ATTRIBUTES ---


def _numeric_task() -> ProcessingTask:
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.data["numeric_attributes"] = ["year", "beats"]
    db_task.save(commit=True)
    return db_task


def _checkbox(body: str, attribute: str) -> str:
    match = re.search(
        rf'<input[^>]*type="checkbox"[^>]*name="pipeline_{re.escape(attribute)}"[^>]*>',
        body,
    )
    assert match, f"no checkbox for {attribute}"
    return match.group(0)


def test_routing_step_frontend_renders_a_checkbox_per_numeric_attribute(client):
    body = client.get(_path("RoutingStepFrontend", db_id=_numeric_task().id)).get_data(
        as_text=True
    )

    for attribute in ("year", "beats"):
        checkbox = _checkbox(body, attribute)
        assert f'value="{INCLUDE_NUMERIC}"' in checkbox
        assert "checked" not in checkbox


def test_routing_step_frontend_restores_the_checked_numeric_attributes(client):
    body = client.post(
        _path("RoutingStepFrontend", db_id=_numeric_task().id),
        data={"pipeline_genre": WU_PALMER_PLUGIN, "pipeline_year": INCLUDE_NUMERIC},
    ).get_data(as_text=True)

    assert "checked" in _checkbox(body, "year")
    assert "checked" not in _checkbox(body, "beats")


def _numeric_settings_task(**payload_overrides) -> ProcessingTask:
    """A task in the routing step with a single-valued and a multi-valued attribute."""
    schema = InputParametersSchema()
    db_task = ProcessingTask(
        task_name="router_test",
        parameters=schema.dumps(schema.load(router_payload(**payload_overrides))),
    )
    db_task.data["taxonomy_attributes"] = []
    db_task.data["numeric_attributes"] = ["year", "beats"]
    db_task.data["multi_valued_numeric_attributes"] = ["beats"]
    db_task.save(commit=True)
    return db_task


def test_routing_step_frontend_renders_the_settings_of_a_multi_valued_attribute(client):
    """A multi-valued numeric attribute runs the mapping, so it needs its settings."""
    db_task = _numeric_settings_task(distanceMetric="cosine", mdsDimensions=3)

    body = client.get(_path("RoutingStepFrontend", db_id=db_task.id)).get_data(
        as_text=True
    )

    for setting in ("distanceMetric", "mdsDimensions", "metric"):
        assert f'name="pipeline_beats__{setting}"' in body
    assert 'value="cosine" selected' in body
    assert 'value="3"' in _tag(body, "pipeline_beats__mds_dimensions")


@pytest.mark.parametrize(
    "setting",
    ["distanceMetric", "mdsDimensions", "rootIsPartOfHierarchy", "transformer"],
)
def test_routing_step_frontend_renders_no_settings_for_a_single_valued_attribute(
    client, setting
):
    """A single-valued numeric attribute is appended to the vector without a plugin."""
    body = client.get(
        _path("RoutingStepFrontend", db_id=_numeric_settings_task().id)
    ).get_data(as_text=True)

    assert f'name="pipeline_year__{setting}"' not in body


def test_routing_step_frontend_renders_a_settings_panel_per_numeric_attribute(client):
    """The panel of a single-valued attribute is a placeholder until it has settings."""
    body = client.get(
        _path("RoutingStepFrontend", db_id=_numeric_settings_task().id)
    ).get_data(as_text=True)

    for attribute in ("year", "beats"):
        assert f'aria-controls="pipeline_{attribute}__settings"' in body
    placeholder = body[body.index('id="pipeline_year__settings"') :]
    assert "no settings yet" in placeholder.split("</div>")[0]


@pytest.mark.parametrize(
    "setting", ["rootIsPartOfHierarchy", "transformer"], ids=["wu_palmer", "transformer"]
)
def test_routing_step_frontend_omits_the_unused_settings_of_a_numeric_attribute(
    client, setting
):
    """Neither Wu-Palmer nor the transformer runs in the numeric mapping pipeline."""
    body = client.get(
        _path("RoutingStepFrontend", db_id=_numeric_settings_task().id)
    ).get_data(as_text=True)

    assert f'name="pipeline_beats__{setting}"' not in body


def test_routing_step_stores_the_settings_of_a_numeric_attribute(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = _numeric_settings_task()

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={
            "pipeline_beats": INCLUDE_NUMERIC,
            "pipeline_beats__distanceMetric": "cosine",
            "pipeline_beats__mdsDimensions": "3",
        },
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    data = ProcessingTask.get_by_id(db_task.id).data
    assert data["attribute_settings"]["beats"]["distanceMetric"] == "cosine"
    assert data["attribute_settings"]["beats"]["mdsDimensions"] == 3


def test_routing_step_accepts_a_numeric_attribute_as_the_only_selection(
    client, monkeypatch
):
    mock_task_dispatch(monkeypatch)
    db_task = _numeric_task()

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={"pipeline_genre": NONE_PLUGIN, "pipeline_year": INCLUDE_NUMERIC},
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert ProcessingTask.get_by_id(db_task.id).data["routing_selections"] == {
        "genre": NONE_PLUGIN,
        "year": INCLUDE_NUMERIC,
    }


def test_routing_step_records_the_base_url_for_the_worker(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = _numeric_task()

    client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={"pipeline_year": INCLUDE_NUMERIC},
    )

    base_url = ProcessingTask.get_by_id(db_task.id).data["base_url"]
    assert base_url == f"http://{current_app.config['SERVER_NAME']}/"
