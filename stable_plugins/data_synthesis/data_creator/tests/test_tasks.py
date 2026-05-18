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

"""End-to-end Celery tests for the data-creator plugin.

The plugin's ``/process/`` endpoint enqueues ``calculation_task`` (see
``routes.py``); these tests exercise the same task through a real
in-process worker on an in-memory broker, mirroring the strategy in
the testing documentation and ADR-0018.
"""

import json
from typing import cast

import pytest
from celery.app.task import Task

from data_creator.backend.datasets import DataTypeEnum
from data_creator.schemas import InputParameters, InputParametersSchema
from data_creator.tasks import calculation_task as _calculation_task_fn
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask

# ``@CELERY.task`` returns a ``Task`` instance at runtime, but the decorator's
# return type is inferred as the wrapped function. Cast once so static analysis
# sees ``apply_async``/``name`` without per-call ``# type: ignore``.
calculation_task = cast(Task, _calculation_task_fn)


def _enqueue_processing_task(params: InputParameters) -> int:
    """Persist a ``ProcessingTask`` the way ``routes.py`` does and return its id."""
    db_task = ProcessingTask(
        task_name=calculation_task.name,  # pyright: ignore[reportArgumentType]
        parameters=InputParametersSchema().dumps(params),
    )
    db_task.save(commit=True)
    return db_task.id


def _outputs_by_name(task: ProcessingTask) -> dict:
    return {output.file_name: output for output in task.outputs}


def _read_json(file_info) -> list:
    """Read a persisted JSON file by its on-disk path.

    ``LocalFileStore`` records the absolute path in ``file_storage_data``,
    so we can read it directly without needing a request context for
    ``url_for``-based URL building.
    """
    with open(file_info.file_storage_data, "r") as fh:
        return json.load(fh)


@pytest.mark.usefixtures("broker_app", "celery_worker")
def test_calculation_task_persists_four_files():
    db_id = _enqueue_processing_task(
        InputParameters(
            dataset_type=DataTypeEnum.checkerboard,
            num_train_points=10,
            num_test_points=5,
        )
    )

    result = calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)
    assert result == "Result stored in file"

    DB.session.expire_all()
    task = ProcessingTask.get_by_id(db_id)
    assert task is not None
    outputs = _outputs_by_name(task)

    expected = {
        "train_data_data-creator_type_checkerboard_amount_10.json": (
            "entity/vector",
            10,
        ),
        "train_labels_data-creator_type_checkerboard_amount_10.json": (
            "entity/label",
            10,
        ),
        "test_data_data-creator_type_checkerboard_amount_5.json": (
            "entity/vector",
            5,
        ),
        "test_labels_data-creator_type_checkerboard_amount_5.json": (
            "entity/label",
            5,
        ),
    }
    assert set(outputs) == set(expected)

    for name, (file_type, expected_count) in expected.items():
        info = outputs[name]
        assert info.file_type == file_type
        assert info.mimetype == "application/json"

        entries = _read_json(info)
        assert len(entries) == expected_count
        first = entries[0]
        # ``set <= dict.keys()`` is a subset check: required keys present, extras allowed.
        assert {"ID", "href"} <= first.keys()
        if file_type == "entity/vector":
            assert {"dim0", "dim1"} <= first.keys()
        else:
            assert "label" in first


@pytest.mark.usefixtures("broker_app", "celery_worker")
def test_calculation_task_with_blobs_passes_centers():
    """``centers`` flows from ``InputParameters`` to ``make_blobs`` via ``**__dict__``."""
    db_id = _enqueue_processing_task(
        InputParameters(
            dataset_type=DataTypeEnum.blobs,
            num_train_points=20,
            num_test_points=0,
            centers=3,
        )
    )

    assert (
        calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)
        == "Result stored in file"
    )

    DB.session.expire_all()
    task = ProcessingTask.get_by_id(db_id)
    assert task is not None
    outputs = _outputs_by_name(task)

    train_data = _read_json(outputs["train_data_data-creator_type_blobs_amount_20.json"])
    test_data = _read_json(outputs["test_data_data-creator_type_blobs_amount_0.json"])
    train_labels = _read_json(
        outputs["train_labels_data-creator_type_blobs_amount_20.json"]
    )

    assert len(train_data) == 20
    assert len(test_data) == 0
    assert {"dim0", "dim1"} <= train_data[0].keys()
    assert "dim2" not in train_data[0]  # 2D blobs, not blobs_3d
    assert {label["label"] for label in train_labels} <= {0, 1, 2}


@pytest.mark.usefixtures("broker_app", "celery_worker")
def test_calculation_task_missing_db_id_raises():
    """Task raises ``KeyError`` when no ``ProcessingTask`` row matches the id."""
    async_result = calculation_task.apply_async(kwargs={"db_id": 99999})
    with pytest.raises(KeyError, match="Could not load task data"):
        async_result.get(timeout=30)
