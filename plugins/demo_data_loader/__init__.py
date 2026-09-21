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

from http import HTTPStatus
from io import BytesIO
from json import dumps, loads
from pathlib import Path
from typing import Mapping, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE, validate

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, SecurityBlueprint
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "demo-data-loader"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


DEMO_DATA_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="A demo dataset for the Feature Engineering Pipeline Plugin.",
)

DATA_ROOT = Path(__file__).parent / "data"

#: Add a new entry here and a matching folder under ``data/`` to ship another dataset.
DATASETS = {
    "animals": "Animals (16 entities, 6 attributes)",
}

TAXONOMIES_DIR_NAME = "taxonomies"


class InputParametersSchema(FrontendFormBaseSchema):
    dataset = ma.fields.String(
        required=True,
        allow_none=False,
        validate=validate.OneOf(DATASETS),
        metadata={
            "label": "Dataset",
            "description": "The demo dataset to load.",
            "input_type": "select",
            "options": DATASETS,
        },
    )


@DEMO_DATA_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @DEMO_DATA_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @DEMO_DATA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Demo data loader",
            description=DemoDataLoader.instance.description,
            name=DemoDataLoader.instance.name,
            version=DemoDataLoader.instance.version,
            type=PluginType.dataloader,
            entry_point=EntryPoint(
                href=url_for(f"{DEMO_DATA_BLP.name}.LoadDemoDataView"),
                ui_href=url_for(f"{DEMO_DATA_BLP.name}.MicroFrontend"),
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="entity/list",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="entity/attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="graph/taxonomy",
                        content_type=["application/zip"],
                        required=True,
                    ),
                ],
            ),
            tags=DemoDataLoader.instance.tags,
        )


@DEMO_DATA_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the demo data loader plugin."""

    @DEMO_DATA_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the demo data loader plugin."
    )
    @DEMO_DATA_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @DEMO_DATA_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @DEMO_DATA_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the demo data loader plugin."
    )
    @DEMO_DATA_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @DEMO_DATA_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=DemoDataLoader.instance.name,
                version=DemoDataLoader.instance.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{DEMO_DATA_BLP.name}.LoadDemoDataView"),
            )
        )


@DEMO_DATA_BLP.route("/process/")
class LoadDemoDataView(MethodView):
    """Start a long running processing task."""

    @DEMO_DATA_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @DEMO_DATA_BLP.response(HTTPStatus.SEE_OTHER)
    @DEMO_DATA_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo data loading task."""
        db_task = ProcessingTask(
            task_name=load_demo_data_task.name, parameters=dumps(arguments)
        )
        db_task.save(commit=True)

        task: chain = load_demo_data_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class DemoDataLoader(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "Loads a small, self contained demo dataset. Provides entities, attribute "
        "metadata and taxonomies that cover tree taxonomies, taxonomies with mapping "
        "vectors and numeric attributes, so that the 'Feature Engineering Pipeline' plugin can "
        "be demonstrated without an external database."
    )
    tags = ["data-loading", "demo"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return DEMO_DATA_BLP


TASK_LOGGER = get_task_logger(__name__)


def build_taxonomies_zip(dataset_dir: Path) -> bytes:
    """Pack every taxonomy of the dataset into a single zip file."""
    taxonomies_dir = dataset_dir / TAXONOMIES_DIR_NAME
    taxonomies = sorted(taxonomies_dir.glob("*.json"))

    if not taxonomies:
        raise FileNotFoundError(
            f"No taxonomy files found in {taxonomies_dir}. The consuming plugins "
            "resolve the 'refTarget' of an attribute against the names in this zip, "
            "so an empty zip silently disables every taxonomy attribute."
        )

    TASK_LOGGER.info(
        f"Packing {len(taxonomies)} taxonomy file(s) from {taxonomies_dir}: "
        f"{[taxonomy.name for taxonomy in taxonomies]}."
    )

    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as zip_file:
        for taxonomy in taxonomies:
            zip_file.writestr(taxonomy.name, taxonomy.read_bytes())

    return zip_buffer.getvalue()


@CELERY.task(name=f"{DemoDataLoader.instance.identifier}.load_demo_data_task", bind=True)
def load_demo_data_task(self, db_id: int) -> str:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    dataset: Optional[str] = loads(task_data.parameters or "{}").get("dataset", None)

    if dataset not in DATASETS:
        raise ValueError(f"Unknown demo dataset '{dataset}'.")

    dataset_dir = DATA_ROOT / dataset
    TASK_LOGGER.info(f"Loading demo dataset '{dataset}' from {dataset_dir}.")

    STORE.persist_task_result(
        db_id,
        (dataset_dir / "entities.json").read_bytes(),
        "entities.json",
        "entity/list",
        "application/json",
    )
    STORE.persist_task_result(
        db_id,
        (dataset_dir / "attribute_metadata.json").read_bytes(),
        "attribute_metadata.json",
        "entity/attribute-metadata",
        "application/json",
    )
    STORE.persist_task_result(
        db_id,
        build_taxonomies_zip(dataset_dir),
        "taxonomies.zip",
        "graph/taxonomy",
        "application/zip",
    )

    return f"Loaded the demo dataset '{dataset}'."
