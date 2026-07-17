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

"""End-to-end Celery tests for the ``mapping_distances`` plugin.

The plugin's ``/process/`` endpoint enqueues ``calculation_task`` (see
``routes.py``). These tests exercise the same task through a real
in-process worker on an in-memory broker, mirroring the strategy in the
testing documentation and ADR-0018.

Input files (entities, attribute metadata, taxonomy zip) are written to a
temporary directory and passed to the task as ``file://`` URLs. The
``FileAdapter`` registered by ``create_app`` lets ``requests`` (and thus
``open_url``) read them, so the worker thread loads them exactly as it
would load remote files in production.
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List
from zipfile import ZipFile

import pytest
from celery.app.task import Task
from typing import cast

from stable_plugins.classical_ml.data_preparation.mapping_distances.schemas import (
    DistanceMetricEnum,
    InputParameters,
    InputParametersSchema,
)
from mapping_distances.tasks import (
    calculation_task as _calculation_task_fn,
)
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask

# ``@CELERY.task`` returns a ``Task`` instance at runtime, but the decorator's
# return type is inferred as the wrapped function. Cast once so static analysis
# sees ``apply_async``/``name`` without per-call ``# type: ignore``.
calculation_task = cast(Task, _calculation_task_fn)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload) -> str:
    """Write ``payload`` as JSON to ``path`` and return a ``file://`` URL."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.as_uri()


def _write_taxonomy_zip(path: Path, taxonomies: Dict[str, dict]) -> str:
    """Write taxonomies as a zip of JSON files and return a ``file://`` URL.

    ``taxonomies`` maps a file name (without ``.json``) to the taxonomy
    JSON content, matching the layout the task expects inside the archive.
    """
    with ZipFile(path, "w") as zip_file:
        for name, content in taxonomies.items():
            zip_file.writestr(f"{name}.json", json.dumps(content))
    return path.as_uri()


def _enqueue_processing_task(params: InputParameters) -> int:
    """Persist a ``ProcessingTask`` the way ``routes.py`` does and return its id."""
    db_task = ProcessingTask(
        task_name=calculation_task.name,  # pyright: ignore[reportArgumentType]
        parameters=InputParametersSchema().dumps(params),
    )
    db_task.save(commit=True)
    return db_task.id


def _read_result_zip(task: ProcessingTask) -> Dict[str, List[dict]]:
    """Read the single zip output and return a mapping ``file name -> entities``."""
    assert len(task.outputs) == 1
    output = task.outputs[0]
    assert output.file_type == "relation/element-distances"
    assert output.mimetype == "application/zip"

    contents: Dict[str, List[dict]] = {}
    with ZipFile(output.file_storage_data, "r") as zip_file:
        for name in zip_file.namelist():
            with zip_file.open(name) as inner:
                contents[name] = json.loads(inner.read().decode("utf-8"))
    return contents


def _build_inputs(tmp_path: Path) -> InputParameters:
    """Create a small, fully deterministic dataset on disk.

    One ``color`` attribute referencing a ``color`` taxonomy with three
    mapped elements; three entities using two distinct colors.
    """
    taxonomy = {
        "entities": [
            {"ID": "red", "mapping": [1.0, 0.0]},
            {"ID": "blue", "mapping": [0.0, 1.0]},
            {"ID": "green", "mapping": [1.0, 1.0]},
        ]
    }
    taxonomies_zip_url = _write_taxonomy_zip(
        tmp_path / "taxonomies.zip", {"color": taxonomy}
    )

    metadata = [
        {
            "ID": "color",
            "type": "color",
            "title": "",
            "description": "ref",
            "multiple": False,
            "ordered": False,
            "separator": ";",
            "refTarget": "taxonomies.zip:color.json",
        }
    ]
    entities_metadata_url = _write_json(tmp_path / "metadata.json", metadata)

    entities = [
        {"ID": "e1", "href": "", "color": "red"},
        {"ID": "e2", "href": "", "color": "blue"},
        {"ID": "e3", "href": "", "color": "red"},
    ]
    entities_url = _write_json(tmp_path / "entities.json", entities)

    return InputParameters(
        entities_url=entities_url,
        entities_metadata_url=entities_metadata_url,
        taxonomies_zip_url=taxonomies_zip_url,
        attributes="color",
        distance_metric=DistanceMetricEnum.euclidean,
    )


def _build_inputs_with_invalid_mapping(tmp_path: Path) -> InputParameters:
    """Dataset where one taxonomy element carries an invalid (NaN) mapping.

    ``blue`` has NaN coordinates, so ``_is_empty_or_nan`` must flag it and the
    task must fall back to ``sys.float_info.max`` for every pair involving it,
    while pairs between valid elements are still computed normally.
    """
    taxonomy = {
        "entities": [
            {"ID": "red", "mapping": [1.0, 0.0]},
            {"ID": "blue", "mapping": [math.nan, math.nan]},
        ]
    }
    taxonomies_zip_url = _write_taxonomy_zip(
        tmp_path / "taxonomies.zip", {"color": taxonomy}
    )

    metadata = [
        {
            "ID": "color",
            "type": "color",
            "title": "",
            "description": "ref",
            "multiple": False,
            "ordered": False,
            "separator": ";",
            "refTarget": "taxonomies.zip:color.json",
        }
    ]
    entities_metadata_url = _write_json(tmp_path / "metadata.json", metadata)

    entities = [
        {"ID": "e1", "href": "", "color": "red"},
        {"ID": "e2", "href": "", "color": "blue"},
    ]
    entities_url = _write_json(tmp_path / "entities.json", entities)

    return InputParameters(
        entities_url=entities_url,
        entities_metadata_url=entities_metadata_url,
        taxonomies_zip_url=taxonomies_zip_url,
        attributes="color",
        distance_metric=DistanceMetricEnum.euclidean,
    )


@pytest.mark.usefixtures("celery_worker")
def test_calculation_task_persists_distance_zip(tmp_path):
    db_id = _enqueue_processing_task(_build_inputs(tmp_path))

    result = calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)
    assert result == "Result stored in file"

    DB.session.expire_all()
    task = ProcessingTask.get_by_id(db_id)
    assert task is not None

    contents = _read_result_zip(task)
    assert set(contents) == {"color.json"}

    distances = {
        (entry["source"], entry["target"]): entry["distance"]
        for entry in contents["color.json"]
    }

    # Two distinct active colors -> a full 2x2 pairwise matrix.
    assert set(distances) == {
        ("blue", "blue"),
        ("blue", "red"),
        ("red", "blue"),
        ("red", "red"),
    }

    # red=[1,0], blue=[0,1] -> euclidean diagonal 0, off-diagonal sqrt(2).
    assert distances[("red", "red")] == pytest.approx(0.0)
    assert distances[("blue", "blue")] == pytest.approx(0.0)
    assert distances[("red", "blue")] == pytest.approx(math.sqrt(2))
    assert distances[("blue", "red")] == pytest.approx(math.sqrt(2))


@pytest.mark.usefixtures("celery_worker")
def test_calculation_task_entry_shape_and_ids(tmp_path):
    db_id = _enqueue_processing_task(_build_inputs(tmp_path))

    calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)

    DB.session.expire_all()
    task = ProcessingTask.get_by_id(db_id)
    assert task is not None

    entries = _read_result_zip(task)["color.json"]
    for entry in entries:
        assert {"source", "target", "distance"} <= entry.keys()


@pytest.mark.usefixtures("celery_worker")
def test_calculation_task_respects_selected_metric(tmp_path):
    params = _build_inputs(tmp_path)
    params.distance_metric = DistanceMetricEnum.manhatten
    db_id = _enqueue_processing_task(params)

    calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)

    DB.session.expire_all()
    task = ProcessingTask.get_by_id(db_id)
    assert task is not None

    # The chosen metric is encoded in the output file name.
    assert "manhatten" in task.outputs[0].file_name.lower()

    distances = {
        (entry["source"], entry["target"]): entry["distance"]
        for entry in _read_result_zip(task)["color.json"]
    }
    # Manhattan distance of red=[1,0] and blue=[0,1] is 2, not sqrt(2).
    assert distances[("red", "blue")] == pytest.approx(2.0)


@pytest.mark.usefixtures("celery_worker")
def test_calculation_task_multiple_attributes(tmp_path):
    color_tax = {
        "entities": [
            {"ID": "red", "mapping": [1.0, 0.0]},
            {"ID": "blue", "mapping": [0.0, 1.0]},
        ]
    }
    size_tax = {
        "entities": [
            {"ID": "small", "mapping": [0.0]},
            {"ID": "large", "mapping": [10.0]},
        ]
    }
    taxonomies_zip_url = _write_taxonomy_zip(
        tmp_path / "taxonomies.zip", {"color": color_tax, "size": size_tax}
    )

    metadata = [
        {
            "ID": "color",
            "type": "color",
            "title": "",
            "description": "ref",
            "multiple": False,
            "ordered": False,
            "separator": ";",
            "refTarget": "taxonomies.zip:color.json",
        },
        {
            "ID": "size",
            "type": "size",
            "title": "",
            "description": "ref",
            "multiple": False,
            "ordered": False,
            "separator": ";",
            "refTarget": "taxonomies.zip:size.json",
        },
    ]
    entities_metadata_url = _write_json(tmp_path / "metadata.json", metadata)

    entities = [
        {"ID": "e1", "href": "", "color": "red", "size": "small"},
        {"ID": "e2", "href": "", "color": "blue", "size": "large"},
    ]
    entities_url = _write_json(tmp_path / "entities.json", entities)

    params = InputParameters(
        entities_url=entities_url,
        entities_metadata_url=entities_metadata_url,
        taxonomies_zip_url=taxonomies_zip_url,
        attributes="color\nsize",
        distance_metric=DistanceMetricEnum.euclidean,
    )
    db_id = _enqueue_processing_task(params)

    calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)

    DB.session.expire_all()
    task = ProcessingTask.get_by_id(db_id)
    assert task is not None

    contents = _read_result_zip(task)
    # One JSON file per requested attribute.
    assert set(contents) == {"color.json", "size.json"}

    size_distances = {
        (entry["source"], entry["target"]): entry["distance"]
        for entry in contents["size.json"]
    }
    assert size_distances[("small", "large")] == pytest.approx(10.0)
    assert size_distances[("large", "large")] == pytest.approx(0.0)

    color_distances = {
        (entry["source"], entry["target"]): entry["distance"]
        for entry in contents["color.json"]
    }
    assert color_distances[("red", "red")] == pytest.approx(0.0)
    assert color_distances[("blue", "blue")] == pytest.approx(0.0)
    assert color_distances[("red", "blue")] == pytest.approx(math.sqrt(2))
    assert color_distances[("blue", "red")] == pytest.approx(math.sqrt(2))


@pytest.mark.usefixtures("celery_worker")
def test_calculation_task_missing_db_id_raises():
    """Task raises ``KeyError`` when no ``ProcessingTask`` row matches the id."""
    async_result = calculation_task.apply_async(kwargs={"db_id": 999999})
    with pytest.raises(KeyError, match="Could not load task data"):
        async_result.get(timeout=30)


@pytest.mark.usefixtures("celery_worker")
def test_calculation_task_flags_invalid_mapping_vectors(tmp_path):
    """A taxonomy element with a NaN mapping is detected by ``_is_empty_or_nan``,
    so every distance involving it falls back to ``sys.float_info.max`` while
    valid pairs are still computed normally.
    """
    db_id = _enqueue_processing_task(_build_inputs_with_invalid_mapping(tmp_path))

    calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)

    DB.session.expire_all()
    task = ProcessingTask.get_by_id(db_id)
    assert task is not None

    distances = {
        (entry["source"], entry["target"]): entry["distance"]
        for entry in _read_result_zip(task)["color.json"]
    }

    # Every pair touching the invalid "blue" mapping falls back to the sentinel.
    assert distances[("blue", "blue")] == sys.float_info.max
    assert distances[("red", "blue")] == sys.float_info.max
    assert distances[("blue", "red")] == sys.float_info.max
    # The valid self-pair is computed normally.
    assert distances[("red", "red")] == pytest.approx(0.0)
