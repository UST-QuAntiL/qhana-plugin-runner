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

from enum import Enum
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, List, Iterator

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE, post_load

from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
    ensure_dict,
)

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    InputDataMetadata,
)
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
    FileUrl,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from qhana_plugin_runner.requests import open_url

from itertools import chain as itr_chain
import numpy as np


_plugin_name = "entity-points-converter"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


CONVERTER_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="entity-points-converter plugin API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class FormatEnum(Enum):
    application_json = "application/json"
    text_csv = "text/csv"


class InputParameters:
    def __init__(self, entities_url: str, output_format: FormatEnum):
        self.entities_url = entities_url
        self.output_format = output_format


class InputParametersSchema(FrontendFormBaseSchema):
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Entity points URL",
            "description": "URL to a json/csv file with the entity points.",
            "input_type": "text",
        },
    )
    output_format = EnumField(
        FormatEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Output Format",
            "description": "The format of the output",
            "input_type": "select",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@CONVERTER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @CONVERTER_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @CONVERTER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=Converter.instance.name,
            description=Converter.instance.description,
            name=Converter.instance.identifier,
            version=Converter.instance.version,
            type=PluginType.conversion,
            entry_point=EntryPoint(
                href=url_for(f"{CONVERTER_BLP.name}.CalcView"),
                ui_href=url_for(f"{CONVERTER_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=["text/csv", "application/json"],
                        required=True,
                        parameter="entitiesUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=["text/csv", "application/json"],
                        required=True,
                    )
                ],
            ),
            tags=Converter.instance.tags,
        )


@CONVERTER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the hello world plugin."""

    @CONVERTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @CONVERTER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @CONVERTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @CONVERTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @CONVERTER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @CONVERTER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=Converter.instance.name,
                version=Converter.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{CONVERTER_BLP.name}.CalcView"),
            )
        )


@CONVERTER_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @CONVERTER_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @CONVERTER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @CONVERTER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=InputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = calculation_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class Converter(QHAnaPluginBase):

    name = _plugin_name
    version = __version__
    description = "Converts a file containing entity points into another file format, e.g. csv into json"
    tags = ["converter"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return CONVERTER_BLP


TASK_LOGGER = get_task_logger(__name__)


def json_to_csv(json_gen, dim_attributes: List[str]) -> Iterator[dict]:
    for ent in json_gen:
        ent_dict = {"ID": ent["ID"], "href": ent["href"]}
        ent_dict.update({d_str: ent["point"][idx] for idx, d_str in enumerate(dim_attributes)})
        yield ent_dict


def get_point(ent, dim_attributes) -> List[float]:
    point = [0.]*len(dim_attributes)
    for idx, d in enumerate(dim_attributes):
        # d is a string dim<int>
        point[idx] = float(ent[d])
    return point


def get_dimensionality(entity):
    return sum(map(lambda key: key.startswith("dim"), entity.keys()))


def get_dim_attributes(dim):
    # for each dimension we have an attribute, i.e. dimension 0 = dim0, dimension 1 = dim1, ...
    # Using the zero padding, we ensure that every dim<int> has the same length, e.g. dim00, dim01, ..., dim10, dim11
    zero_padding = len(str(dim - 1))
    dim_attributes = [f"dim{str(d).rjust(zero_padding, '0')}" for d in range(dim)]
    return dim_attributes


def csv_to_json_gen(file_, file_type) -> Iterator[dict]:
    gen = load_entities(file_, mimetype=file_type)
    gen = ensure_dict(gen)

    first_ent = next(gen)
    gen = itr_chain((first_ent, ), gen)
    dim = get_dimensionality(first_ent)
    dim_attributes = get_dim_attributes(dim)


    # Yield all elements once in json format
    for ent in gen:
        yield {"ID": ent["ID"], "href": ent["href"], "point": get_point(ent, dim_attributes)}


def json_to_json_gen(file_, file_type) -> Iterator[dict]:
    # Yield all elements once in json format
    return load_entities(file_, mimetype=file_type)


def get_dim_and_json_gen(file_) -> (int, Iterator[dict]):
    header = file_.headers["Content-Type"]
    # Header might contain more than just the input format
    header = header.split(';')
    input_format = header[0]

    # If header contains the encoding type (charset), then take it. Otherwise, set it to utf-8
    encoding = "utf-8"
    for i in range(len(header)):
        if "charset=" in header[i]:
            encoding = header[i].split("charset=")[1]
    file_.encoding = encoding

    # Check input format and convert it to json format
    # Generators provided here, must include the first entity twice, to extract the dimensionality from it!
    if input_format == "text/csv":
        gen = csv_to_json_gen(file_, input_format)
    elif input_format == "application/json":
        gen = json_to_json_gen(file_, input_format)
    else:
        raise ValueError(f"Converting input format {input_format} is not implemented!")

    # Check the dimensionality of the points.
    # Since all points should have the same dimensionality, checking the first point suffices.
    first_ent = next(gen)
    dim = len(first_ent["point"])

    # Gen should contain all elements
    gen = itr_chain((first_ent,), gen)

    return dim, gen


def get_output(entities_url, output_format) -> (List[str], Iterator[dict], str):
    """
    This method first converts every input into the json format and from that it converts it to the specified output
    format. This saves converters, but there are always two converters involved.
    """
    file_ = open_url(entities_url, stream=True)
    # First we load in the data and convert it into json format
    dim, json_gen = get_dim_and_json_gen(file_)

    # Since everything is in json format, we can now convert it to the desired output format
    if output_format == "text/csv":
        # Ensure that all dim<int> attributes have the same length, e.g. dim00, dim01, ..., dim10, dim11
        # instead of dim0, dim1, ..., dim10, dim11
        dim_attributes = get_dim_attributes(dim)
        attributes = ["ID", "href"] + dim_attributes
        return attributes, json_to_csv(json_gen, dim_attributes), ".csv"
    elif output_format == "application/json":
        return [], json_gen, ".json"
    else:
        raise ValueError(
            f"Converting into output format {output_format} is not implemented!"
        )


@CELERY.task(name=f"{Converter.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new one-hot calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entities_url = input_params.entities_url
    TASK_LOGGER.info(f"Loaded input parameters from db: entities_url='{entities_url}'")
    output_format = input_params.output_format.value
    TASK_LOGGER.info(f"Loaded input parameters from db: output_format='{output_format}'")

    attributes, entity_points, file_ending = get_output(entities_url, output_format)
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entity_points, output, output_format, attributes=attributes)
        STORE.persist_task_result(
            db_id,
            output,
            "converted_entity_points" + file_ending,
            "entity/vector",
            output_format,
        )

    return "Result stored in file"