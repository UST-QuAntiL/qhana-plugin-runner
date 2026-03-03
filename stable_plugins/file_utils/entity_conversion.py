# Copyright 2021 QHAna plugin runner contributors.
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
from itertools import chain as chain_iter
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import abort, redirect
from flask.app import Flask
from flask.globals import current_app, request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import (
    AttributeMetadata,
    dict_serializer,
    tuple_deserializer,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    EntityTupleMixin,
    entity_attribute_sort_key,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.requests import (
    get_mimetype,
    open_url,
    retrieve_attribute_metadata_url,
    retrieve_data_type,
    retrieve_filename,
)
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_csv_plugin_name = "csv-to-json"
_json_plugin_name = "json-to-csv"
__version__ = "v0.1.0"
_csv_identifier = plugin_identifier(_csv_plugin_name, __version__)
_json_identifier = plugin_identifier(_json_plugin_name, __version__)


CSV_TO_JSON_BLP = SecurityBlueprint(
    _csv_identifier,  # blueprint name
    __name__,  # module import name!
    description="Convert entities from csv to json.",
)


JSON_TO_CSV_BLP = SecurityBlueprint(
    _json_identifier,  # blueprint name
    __name__,  # module import name!
    description="Convert entities from json to CSV.",
)


class CSVInputSchema(FrontendFormBaseSchema):
    data = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="*",
        data_content_types=["text/csv"],
        metadata={
            "label": "Data URL",
            # TODO: check "input_type": "url",
        },
    )


class JSONInputSchema(FrontendFormBaseSchema):
    data = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="*",
        data_content_types=["application/json"],
        metadata={
            "label": "Data URL",
            # TODO: check "input_type": "url",
        },
    )


@CSV_TO_JSON_BLP.route("/")
class CsvPluginView(MethodView):
    """Root resource of the csv to json plugin."""

    @CSV_TO_JSON_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @CSV_TO_JSON_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = CsvToJson.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.conversion,
            entry_point=EntryPoint(
                href=url_for(f"{CSV_TO_JSON_BLP.name}.{CsvProcessView.__name__}"),
                ui_href=url_for(f"{CSV_TO_JSON_BLP.name}.{CsvMicroFrontend.__name__}"),
                plugin_dependencies=[],
                data_input=[
                    InputDataMetadata(
                        data_type="*",
                        content_type=["text/csv"],
                        required=True,
                        parameter="data",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="*",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
            ),
            tags=plugin.tags,
        )


@JSON_TO_CSV_BLP.route("/")
class JsonPluginView(MethodView):
    """Root resource of the json to csv plugin."""

    @JSON_TO_CSV_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @JSON_TO_CSV_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = JsonToCsv.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.conversion,
            entry_point=EntryPoint(
                href=url_for(f"{JSON_TO_CSV_BLP.name}.{JsonProcessView.__name__}"),
                ui_href=url_for(f"{JSON_TO_CSV_BLP.name}.{JsonMicroFrontend.__name__}"),
                plugin_dependencies=[],
                data_input=[
                    InputDataMetadata(
                        data_type="*",
                        content_type=["application/json"],
                        required=True,
                        parameter="data",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="*",
                        content_type=["text/csv"],
                        required=True,
                    )
                ],
            ),
            tags=plugin.tags,
        )


@CSV_TO_JSON_BLP.route("/ui/")
class CsvMicroFrontend(MethodView):
    """Micro frontend for the csv to json plugin."""

    @CSV_TO_JSON_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the csv to json plugin."
    )
    @CSV_TO_JSON_BLP.arguments(
        CSVInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @CSV_TO_JSON_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @CSV_TO_JSON_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the csv to json plugin."
    )
    @CSV_TO_JSON_BLP.arguments(
        CSVInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @CSV_TO_JSON_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = CsvToJson.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = CSVInputSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{CSV_TO_JSON_BLP.name}.{CsvProcessView.__name__}"),
                example_values=url_for(
                    f"{CSV_TO_JSON_BLP.name}.{CsvMicroFrontend.__name__}"
                ),
            )
        )


@JSON_TO_CSV_BLP.route("/ui/")
class JsonMicroFrontend(MethodView):
    """Micro frontend for the json to csv plugin."""

    @JSON_TO_CSV_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the json to csv plugin."
    )
    @JSON_TO_CSV_BLP.arguments(
        JSONInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @JSON_TO_CSV_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @JSON_TO_CSV_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the json to csv plugin."
    )
    @JSON_TO_CSV_BLP.arguments(
        JSONInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @JSON_TO_CSV_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = CsvToJson.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = JSONInputSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{JSON_TO_CSV_BLP.name}.{JsonProcessView.__name__}"),
                example_values=url_for(
                    f"{JSON_TO_CSV_BLP.name}.{JsonMicroFrontend.__name__}"
                ),
            )
        )


@CSV_TO_JSON_BLP.route("/process/")
class CsvProcessView(MethodView):
    """Start a long running processing task."""

    @CSV_TO_JSON_BLP.arguments(CSVInputSchema(unknown=EXCLUDE), location="form")
    @CSV_TO_JSON_BLP.response(HTTPStatus.SEE_OTHER)
    @CSV_TO_JSON_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the conversion task."""
        db_task = ProcessingTask(task_name="convert_csv", parameters=arguments["data"])
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = convert_csv.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@JSON_TO_CSV_BLP.route("/process/")
class JsonProcessView(MethodView):
    """Start a long running processing task."""

    @JSON_TO_CSV_BLP.arguments(JSONInputSchema(unknown=EXCLUDE), location="form")
    @JSON_TO_CSV_BLP.response(HTTPStatus.SEE_OTHER)
    @JSON_TO_CSV_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the conversion task."""
        db_task = ProcessingTask(task_name="convert_json", parameters=arguments["data"])
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = convert_csv.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class CsvToJson(QHAnaPluginBase):
    name = _csv_plugin_name
    version = __version__
    description = "Convert CSV files to JSON."
    tags = ["conversion", "csv", "json"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return CSV_TO_JSON_BLP


class JsonToCsv(QHAnaPluginBase):
    name = _json_plugin_name
    version = __version__
    description = "Convert JSON files to CSV."
    tags = ["conversion", "csv", "json"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return JSON_TO_CSV_BLP


TASK_LOGGER = get_task_logger(__name__)


def _fill_in_metadata(
    metadata: dict[str, AttributeMetadata], data_type: str, attributes: tuple[str, ...]
) -> dict[str, AttributeMetadata]:
    if "ID" not in metadata:
        metadata["ID"] = AttributeMetadata("ID", "string", "ID")
    if "href" not in metadata:
        metadata["href"] = AttributeMetadata("href", "string", "href")

    if not (metadata.keys() <= {"ID", "href"}):
        return metadata

    if data_type in (
        "entity/numeric",
        "entity/vector",
        "entity/shaped_vector",
        "entity/matrix",
    ):
        for attr in attributes:
            if attr not in ("ID", "href"):
                metadata[attr] = AttributeMetadata(attr, "number", attr)

    if data_type == "entity/label":
        metadata["label"] = AttributeMetadata("label", "string", "label")

    return metadata


def _convert_data(db_task: ProcessingTask):
    data_url = db_task.parameters

    with open_url(data_url) as entities_data, SpooledTemporaryFile(mode="w") as output:
        mimetype = get_mimetype(entities_data)
        filename = retrieve_filename(entities_data)
        data_type = retrieve_data_type(entities_data)
        metadata_url = retrieve_attribute_metadata_url(entities_data)
        entities_metadata: dict[str, AttributeMetadata] = {}

        if metadata_url:
            with open_url(metadata_url) as entities_metadata_file:
                entities_metadata_list = list(
                    load_entities(
                        entities_metadata_file, get_mimetype(entities_metadata_file)
                    )
                )
                entities_metadata = {
                    element["ID"]: AttributeMetadata.from_dict(element)
                    for element in entities_metadata_list
                }

        def load():
            nonlocal entities_metadata
            deserializer = None
            ent_attributes: tuple[str, ...] | None = None
            for ent in load_entities(entities_data, mimetype):
                if isinstance(ent, EntityTupleMixin):  # is NamedTuple
                    if deserializer is None:
                        ent_attributes = type(ent).entity_attributes
                        entities_metadata = _fill_in_metadata(
                            entities_metadata, data_type, ent_attributes
                        )
                        ent_tuple = type(ent)
                        deserializer = tuple_deserializer(
                            ent_attributes, entities_metadata, tuple_=ent_tuple._make
                        )

                    ent = deserializer(ent)
                    yield ent.as_dict()
                else:
                    if not ent_attributes:
                        ent_attributes = tuple(ent.keys())
                    yield ent

        entities = load()
        first = next(entities)
        ent_attributes = tuple(sorted(first.keys(), key=entity_attribute_sort_key))
        entities = chain_iter([first], entities)

        save_mimetype: str

        if mimetype == "application/json":
            serializer = dict_serializer(ent_attributes, entities_metadata)
            entities = (serializer(e) for e in entities)
            save_mimetype = "text/csv"
            filename += ".csv"
            save_entities(entities, output, save_mimetype, ent_attributes)
        elif mimetype == "text/csv":
            save_mimetype = "application/json"
            filename += ".json"
            save_entities(entities, output, save_mimetype, ent_attributes)
        else:
            raise ValueError(f"Unsupported mimetype '{mimetype}'.")

        if data_type is None:
            # assume default data type
            data_type = "entity/list"

        STORE.persist_task_result(
            db_task.id,
            output,
            filename,
            data_type,
            save_mimetype,
        )


@CELERY.task(name=f"{CsvToJson.instance.identifier}.convert_csv", bind=True)
def convert_csv(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new entity data conversion task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    _convert_data(task_data)

    return "Result stored in file"


@CELERY.task(name=f"{JsonToCsv.instance.identifier}.convert_json", bind=True)
def convert_json(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new entity data conversion task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    _convert_data(task_data)

    return "Result stored in file"
