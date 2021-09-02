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

from enum import Enum
import mimetypes
from plugins.costume_loader_pkg.schemas import InputParameters
import random
from http import HTTPStatus
from json import dumps, loads, JSONEncoder
from qhana_plugin_runner.plugin_utils.entity_marshalling import ensure_dict, load_entities, save_entities
from qhana_plugin_runner.requests import open_url
from typing import Dict, Mapping, Optional

import marshmallow as ma
from qhana_plugin_runner.api.extra_fields import EnumField
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from sqlalchemy.sql.expression import select

from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from tempfile import SpooledTemporaryFile
from qhana_plugin_runner.storage import STORE
from flask import redirect

_plugin_name = "costume-filter"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


COSTUME_FILTER_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Costume filter API.",
    template_folder="costume_filter_templates",
)


class ResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class AttributeType(Enum):
    ALLOWLIST = "Allowlist"
    BLOCKLIST = "Blocklist"


class RowSamplingType(Enum):
    RANDOM = "Randomly"
    FIRST_N = "First n (Number of Rows)"


class EnumEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return JSONEncoder.default(self, obj)


class CostumeFilterParametersSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True, allow_none=False, load_only=True, metadata={"label": "Input Data"}
    )

    attributes = ma.fields.String( # TODO: maybe via ma.fields.List(ma.fields.String(),
        required=False,
        allow_none=True,
        metadata={
            "label": "Attribute List",
            "description": "List of attributes in allowlist/blocklist.", 
            "input_type": "textarea",
        }, # TODO: validation comma separated list => Phillip fragen... nur gegen input_data 
    )

    attributes_setting = EnumField(
        AttributeType,
        required=True,
        metadata={
            "label": "Attributes Setting",
            "description": "Specify attribute list as allowlist or blocklist.", 
            "input_type": "select",
        },
    )

    n_rows = ma.fields.Integer(
        required=False, 
        allow_none=True,
        metadata={
            "label": "Number of Rows",
            "description": "Integer of number of rows that should be kept.",
            "input_type": "textfield",
        },
    )
    
    row_sampling = EnumField( 
        RowSamplingType,
        required=True,
        metadata={
            "label": "Row Sampling",
            "description": "Specify if rows are chosen randomly or first n in case that Number \
                of Rows is set and ID list smaller than Number of Rows.",
            "input_type": "select",
        },
    )

    id_list = ma.fields.String( 
        required=False,
        allow_none=True,
        metadata={
            "label": "ID List",
            "description": "Comma separated list of ID's that should be kept. If number is \
                smaller than Number of Rows, remaining rows are chosen according to Row Choice.",
            "input_type": "textarea",
        },
    )


@COSTUME_FILTER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @COSTUME_FILTER_BLP.response(HTTPStatus.OK, ResponseSchema())
    @COSTUME_FILTER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Costume filter endpoint returning the plugin metadata."""
        return {
            "name": CostumeFilter.instance.name,
            "version": CostumeFilter.instance.version,
            "identifier": CostumeFilter.instance.identifier,
            "root_href": url_for(f"{COSTUME_FILTER_BLP.name}.PluginsView"),
            "title": "Costume loader",
            "description": "Filters data sets from the MUSE database.",
            "plugin_type": "data-loader",
            "tags": ["data:loading"],
            "processing_resource_metadata": {
                "href": url_for(f"{COSTUME_FILTER_BLP.name}.ProcessView"),
                "ui_href": url_for(f"{COSTUME_FILTER_BLP.name}.MicroFrontend"),
                "inputs": [ # TODO: only file input (entities...)
                    [
                        {
                            "output_type": "raw", 
                            "content_type": "application/json",
                            "name": "Raw costume data",
                        },
                        {
                            "output_type": "raw",
                            "content_type": "text/csv",
                            "name": "Raw costume data",
                        },
                        # TODO: OR -> json, csv... scatch, not finalized yet
                    ]
                ],
                "outputs": [
                    [
                        { # TODO: file handle to filtered file, could be json or csv...
                            "output_type": "raw",
                            "content_type": "application/json",
                            "name": "Filtered raw costume data",
                        },
                    ]
                ],
            },
        }


@COSTUME_FILTER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the costume filter plugin."""

    example_inputs = {
        "n_rows": 20,
        "row_choice": "random",
    }

    @COSTUME_FILTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the costume filter plugin."
    )
    @COSTUME_FILTER_BLP.arguments(
        CostumeFilterParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @COSTUME_FILTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @COSTUME_FILTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the costume filter plugin."
    )
    @COSTUME_FILTER_BLP.arguments(
        CostumeFilterParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @COSTUME_FILTER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = CostumeFilterParametersSchema()
        return Response(
            render_template(
                "costume_filter_template.html",
                name=CostumeFilter.instance.name,
                version=CostumeFilter.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{COSTUME_FILTER_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{COSTUME_FILTER_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@COSTUME_FILTER_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @COSTUME_FILTER_BLP.arguments(CostumeFilterParametersSchema(unknown=EXCLUDE), location="form")
    @COSTUME_FILTER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @COSTUME_FILTER_BLP.require_jwt("jwt", optional=True)
    def post(self, input_params : InputParameters):
        """Start the costume filter task."""
        db_task = ProcessingTask(
            task_name=costume_filter_task.name, 
            parameters=dumps(input_params, cls=EnumEncoder))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = costume_filter_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(result.id)), HTTPStatus.SEE_OTHER
        )


class CostumeFilter(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return COSTUME_FILTER_BLP
    
    def get_requirements(self) -> str:
        return ""


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{CostumeFilter.instance.identifier}.costume_filter_task", bind=True)
def costume_filter_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new costume filter task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params : Dict = loads(task_data.parameters or "{}")
    input_file_url: Optional[str] = params.get("input_file_url", None)
    attributes: Optional[str] = params.get("attributes", None)
    attributes_setting: Optional[str] = params.get("attributes_setting", None)
    n_rows: Optional[int] = params.get("n_rows", None)
    row_sampling : Optional[str] = params.get("row_sampling", None)
    id_list: Optional[str] = params.get("id_list", None)
    
    TASK_LOGGER.info(f"Loaded input parameters from db: input_file_url='{input_file_url}', \
        attributes='{attributes}', attributes_setting='{attributes_setting}', n_rows='{n_rows}', \
        row_sampling='{row_sampling}', id_list='{id_list}'")

    if input_file_url is None or not input_file_url:
        raise ValueError("No input file URL provided!")
   
    if not attributes: # empty string or None
        attributes = []
    else:
        attributes = [a.strip() for a in attributes.split(',')] # TODO: raise exception if format not correct?
        
    if n_rows is None and not id_list:
        raise ValueError("Row number argument not provided!")
    if not id_list:
        id_list = []
        n_sampled_rows = n_rows
    else:
        id_list = [id.strip() for id in id_list.split(',')]
        if n_rows is None:
            n_rows = len(id_list)
        n_sampled_rows = max(0, n_rows - len(id_list)) 

    if len(id_list) > n_rows:
        raise ValueError("Length of ID list greater than number of rows!")

    if row_sampling is None and n_sampled_rows > 0:
        raise ValueError("Row sampling not specified!")

    if attributes_setting is None:
        raise ValueError("Attribute setting not specified!")
    
    # Filter rows: get requested entities from input file 
    output_entities_id_list = []
    # Random sampling as in https://math.stackexchange.com/questions/846036/can-i-uniformly-sample-from-n-distinct-elements-where-n-is-unknown-but-fini
    output_entities_random_rows = []
    sampling_counter = 0
    mimetype = mimetypes.MimeTypes().guess_type(input_file_url)[0]
    with open_url(input_file_url, stream=True) as url_data:
        for entity in ensure_dict(load_entities(file_=url_data, mimetype=mimetype)):
            if entity["ID"] in id_list:
                # find entities in id_list if id_list not empty
                output_entities_id_list.append(entity) # TODO
                id_list.remove(entity["ID"])

            elif n_sampled_rows > 0:
                # sample rows to fill up remaining rows according to row sampling
                if row_sampling == RowSamplingType.RANDOM.value:
                    if sampling_counter < n_sampled_rows:
                        # add first n
                        output_entities_random_rows.append(entity)
                        sampling_counter += 1
                    else:
                        # add with prob n/(n+1) at random index
                        if random.random() < n_sampled_rows/(sampling_counter + 1):
                            
                            index = random.randrange(n_sampled_rows)
                            TASK_LOGGER.info(n_sampled_rows/(sampling_counter + 1))
                            TASK_LOGGER.info("Index: " + str(index) + ", ID: " + entity["ID"])
                            output_entities_random_rows[index] = entity
                        sampling_counter += 1
                elif row_sampling == RowSamplingType.FIRST_N.value:
                    if sampling_counter < n_sampled_rows:
                        output_entities_random_rows.append(entity)
                        sampling_counter += 1
                else:
                    raise ValueError("Invalid argument for Row Sampling!")

        if id_list: # not all ID's in file
            raise ValueError(f"The following ID's could not be found: {str(id_list)}")
            # TODO: do sth else?

    output_entities = output_entities_id_list + output_entities_random_rows
    if len(output_entities) != n_rows:
        raise ValueError("Number of rows requested is greater than number of rows in input file!")
        
    # Filter columns
    # make sure that "ID" is not deleted
    if attributes_setting == AttributeType.ALLOWLIST.value:
        attributes = attributes + ["ID"]
    elif attributes_setting == AttributeType.BLOCKLIST.value:
        if "ID" in attributes:
            attributes = attributes.remove("ID")
    else:
        raise ValueError("Invalid argument for Attribute Setting!")

    # remove columns that are not in allowlist
    for entity in output_entities:
        if attributes_setting == AttributeType.ALLOWLIST.value:
            for attr in entity.copy().keys():
                if attr not in attributes:
                    del entity[attr]
        else: # Blocklist
            for attr in entity.copy().keys():
                if attr in attributes:
                    del entity[attr]

        TASK_LOGGER.info(str(entity))
    

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entities=output, file_=output, mimetype=mimetype) # TODO: check

        if mimetype == "application/json":
            file_type = ".json"
        else:
            file_type = ".csv"
        STORE.persist_task_result(
            db_id, output, "filtered_costumes" + file_type, "costume_filter_output", mimetype
        ) # TODO: check
    return "Filter successful." # TODO
    