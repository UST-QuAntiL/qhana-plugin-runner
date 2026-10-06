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

"""Checks that the shipped demo data matches what the consuming plugins expect."""

import json
import zipfile
from pathlib import PurePath

import pytest

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import NUMERIC_TYPES, AttributeMetadata
from tests.utils import run_task

from .. import DATA_ROOT, DATASETS, TAXONOMIES_DIR_NAME, load_demo_data_task

DATASET_NAMES = sorted(DATASETS)


def _read_json(*path_parts):
    return json.loads((DATA_ROOT.joinpath(*path_parts)).read_text(encoding="utf-8"))


def _metadata(dataset):
    return [
        AttributeMetadata.from_dict(element)
        for element in _read_json(dataset, "attribute_metadata.json")
    ]


def _taxonomy(dataset, file_name):
    return _read_json(dataset, TAXONOMIES_DIR_NAME, file_name)


def _values(entity, attribute):
    value = entity.get(attribute)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_every_attribute_is_described_and_used(dataset):
    entities = _read_json(dataset, "entities.json")
    described = {meta.ID for meta in _metadata(dataset)}

    for entity in entities:
        attributes = set(entity) - {"ID"}
        assert attributes == described


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_every_taxonomy_reference_resolves(dataset):
    entities = _read_json(dataset, "entities.json")

    for meta in _metadata(dataset):
        if not meta.ref_target:
            continue

        file_name = PurePath(meta.ref_target.split(":")[1]).name
        taxonomy = _taxonomy(dataset, file_name)
        known_items = {node["ID"] for node in taxonomy["entities"]}

        for entity in entities:
            for value in _values(entity, meta.ID):
                assert value in known_items, (
                    f"Entity '{entity['ID']}' references the unknown taxonomy item "
                    f"'{value}' in attribute '{meta.ID}'."
                )


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_every_taxonomy_relation_connects_known_items(dataset):
    for taxonomy_file in (DATA_ROOT / dataset / TAXONOMIES_DIR_NAME).glob("*.json"):
        taxonomy = json.loads(taxonomy_file.read_text(encoding="utf-8"))
        known_items = {node["ID"] for node in taxonomy["entities"]}

        for relation in taxonomy["relations"]:
            assert relation["source"] in known_items
            assert relation["target"] in known_items


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_mapping_taxonomies_map_every_item_to_the_same_dimension(dataset):
    """A node without a mapping would produce the maximum distance for that item."""
    for taxonomy_file in (DATA_ROOT / dataset / TAXONOMIES_DIR_NAME).glob("*.json"):
        taxonomy = json.loads(taxonomy_file.read_text(encoding="utf-8"))
        dimensions = {len(node["mapping"]) for node in taxonomy["entities"]}

        assert len(dimensions) == 1, (
            f"Taxonomy '{taxonomy_file.name}' mixes mapped and unmapped items "
            f"or mappings of different lengths: {sorted(dimensions)}."
        )

        for node in taxonomy["entities"]:
            parsed = [float(part) for part in node["mapping_raw"].split()]
            assert parsed == node["mapping"], (
                f"Item '{node['ID']}' in '{taxonomy_file.name}' has a 'mapping_raw' "
                "that does not match its parsed 'mapping'."
            )


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_numeric_attributes_are_complete_and_numeric(dataset):
    """Incomplete numeric columns make the feature engineering pipeline fail."""
    entities = _read_json(dataset, "entities.json")

    for meta in _metadata(dataset):
        if meta.description not in NUMERIC_TYPES:
            continue

        for entity in entities:
            values = _values(entity, meta.ID)
            assert values, (
                f"Entity '{entity['ID']}' has no value for the numeric attribute "
                f"'{meta.ID}'."
            )
            assert all(isinstance(value, (int, float)) for value in values)

        if meta.multiple:
            lengths = {len(_values(entity, meta.ID)) for entity in entities}
            assert len(lengths) == 1, (
                f"Attribute '{meta.ID}' mixes value counts {sorted(lengths)}, which "
                "would be zero padded to the longest vector."
            )


@pytest.mark.parametrize("dataset", DATASET_NAMES)
@pytest.mark.usefixtures("celery_worker")
def test_the_task_stores_the_three_pipeline_inputs(dataset):
    db_task = ProcessingTask(
        task_name=load_demo_data_task.name, parameters=json.dumps({"dataset": dataset})
    )
    db_task.save(commit=True)

    run_task(load_demo_data_task, db_id=db_task.id)  # pyright: ignore[reportArgumentType]

    task = ProcessingTask.get_by_id(db_task.id)
    assert task is not None
    outputs = {output.file_type: output for output in task.outputs}

    assert set(outputs) == {
        "entity/list",
        "entity/attribute-metadata",
        "graph/taxonomy",
    }
    assert outputs["entity/list"].mimetype == "application/json"
    assert outputs["entity/attribute-metadata"].mimetype == "application/json"
    assert outputs["graph/taxonomy"].mimetype == "application/zip"

    expected_taxonomies = {
        PurePath(meta.ref_target.split(":")[1]).name
        for meta in _metadata(dataset)
        if meta.ref_target
    }
    with zipfile.ZipFile(outputs["graph/taxonomy"].file_storage_data) as archive:
        assert expected_taxonomies <= set(archive.namelist())
