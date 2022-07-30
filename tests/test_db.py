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

from conftests import task_data
from sqlalchemy_json import TrackedDict, TrackedList

from qhana_plugin_runner.db.models.tasks import ProcessingTask


def test_mutable_json_init(task_data: ProcessingTask):
    assert (
        task_data.data == dict()
    ), "Task data.data is expected to be an empty dict at the start of the unit test."


def test_mutable_json_primitive_values(task_data: ProcessingTask):
    task_data.data = None
    task_data.save(commit=True)
    assert isinstance(
        task_data.data, type(None)
    ), "Task data.data is expected to be of type NoneType."

    task_data.data = bool(True)
    task_data.save(commit=True)
    assert isinstance(
        task_data.data, bool
    ), "Task data.data is expected to be of type bool."
    assert (
        task_data.data == True
    ), "Failed to persist a change to a boolean value to the database."

    task_data.data = 1
    task_data.save(commit=True)
    assert isinstance(
        task_data.data, int
    ), "Task data.data is expected to be of type int."
    assert (
        task_data.data == 1
    ), "Failed to persist a change to an int value to the database."

    task_data.data = "Test"
    task_data.save(commit=True)
    assert isinstance(
        task_data.data, str
    ), "Task data.data is expected to be of type str."
    assert (
        task_data.data == "Test"
    ), "Failed to persist a change to a string value to the database."

    task_data.data = 1.12
    task_data.save(commit=True)
    assert isinstance(
        task_data.data, float
    ), "Task data.data is expected to be of type float."
    assert (
        task_data.data == 1.12
    ), "Failed to persist a change to a float value to the database."


def test_mutable_json_nested(task_data: ProcessingTask):
    task_data.data = {}
    task_data.data["test"] = {"x": 1, "y": 2}
    task_data.save(commit=True)
    assert isinstance(
        task_data.data, TrackedDict
    ), "Task data.data is expected to be of type TackedDict."
    assert isinstance(
        task_data.data["test"], TrackedDict
    ), "Task data.data['test'] is expected to be of type TrackedDict."
    assert (
        task_data.data["test"]["y"] == 2
    ), "Failed to correctly persist a nested dict to the database."

    task_data.data["test"]["x"] += 10
    task_data.save(commit=True)
    assert (
        task_data.data["test"]["x"] == 11
    ), "Failed to persist a change to a nested dict entry to the database."

    task_data.data["test"]["x"] = []
    task_data.save(commit=True)
    assert (
        task_data.data["test"]["x"] == []
    ), "Failed to persist a change to a nested dict entry to the database (assignment of nested list)."

    task_data.data = []
    task_data.data.append({"x": 1, "y": 2})
    task_data.save(commit=True)
    assert isinstance(
        task_data.data, TrackedList
    ), "Task data.data is expected to be of type TrackedList."
    assert isinstance(
        task_data.data[0], TrackedDict
    ), "Task data.data[0] is expected to be of type TrackedDict."
    assert (
        task_data.data[0]["x"] == 1
    ), "Failed to persist a change to an entry of a nested dict within a nested list to the database."

    task_data.data[0] = 1
    task_data.save(commit=True)
    assert (
        task_data.data[0] == 1
    ), "Failed to persist a change to a nested list entry to the database."
