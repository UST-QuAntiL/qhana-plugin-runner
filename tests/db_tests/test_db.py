from conftests import task_data

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.mutable_json import (
    TrackedDict,
    TrackedList,
)


def test_json_strorage(task_data: ProcessingTask):
    # Test mutable json data attribute
    assert task_data.data == dict()

    # Primitive data types
    task_data.data = bool(True)
    task_data.save(commit=True)
    assert isinstance(task_data.data, bool)
    assert task_data.data == True

    task_data.data = 1
    task_data.save(commit=True)
    assert isinstance(task_data.data, int)
    assert task_data.data == 1

    task_data.data = "Test"
    task_data.save(commit=True)
    assert isinstance(task_data.data, str)
    assert task_data.data == "Test"

    task_data.data = 1.12
    task_data.save(commit=True)
    assert isinstance(task_data.data, float)
    assert task_data.data == 1.12

    # Nested data types
    task_data.data = {}
    task_data.data["test"] = {"x": 1, "y": 2}
    task_data.save(commit=True)
    assert isinstance(task_data.data, TrackedDict)
    assert isinstance(task_data.data["test"], TrackedDict)
    assert task_data.data["test"]["y"] == 2

    task_data.data["test"]["x"] += 10
    task_data.save(commit=True)
    assert task_data.data["test"]["x"] == 11

    task_data.data["test"]["x"] = []
    task_data.save(commit=True)
    assert task_data.data["test"]["x"] == []

    task_data.data = []
    task_data.data.append({"x": 1, "y": 2})
    task_data.save(commit=True)
    assert isinstance(task_data.data, TrackedList)
    assert isinstance(task_data.data[0], TrackedDict)
    assert task_data.data[0]["x"] == 1

    task_data.data[0] = 1
    task_data.save(commit=True)
    assert task_data.data[0] == 1
