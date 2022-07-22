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
import json
from enum import Enum
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response
from flask import redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE, post_load

from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
    InputDataMetadata,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
    FileUrl,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "distance-aggregator"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


AGGREGATOR_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Distance aggregator plugin API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class AggregatorsEnum(Enum):
    mean = "Mean"
    median = "Median"
    max = "Max"
    min = "Min"


class InputParameters:
    def __init__(
        self,
        attribute_distances_url: str,
        aggregator: AggregatorsEnum,
    ):
        self.attribute_distances_url = attribute_distances_url
        self.aggregator = aggregator


class InputParametersSchema(FrontendFormBaseSchema):
    attribute_distances_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="attribute-distances",
        data_content_types="application/zip",
        metadata={
            "label": "Attribute distances URL",
            "description": "URL to a zip file with the attribute distances.",
            "input_type": "text",
        },
    )
    aggregator = EnumField(
        AggregatorsEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Aggregator",
            "description": "Aggregator that shall be used to aggregate the attribute distances to a single distance value.",
            "input_type": "select",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@AGGREGATOR_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @AGGREGATOR_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @AGGREGATOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Aggregators endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Aggregators",
            name=Aggregator.instance.name,
            description=Aggregator.instance.description,
            version=Aggregator.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{AGGREGATOR_BLP.name}.CalcSimilarityView"),
                ui_href=url_for(f"{AGGREGATOR_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="attribute-distances",
                        content_type=["application/zip"],
                        required=True,
                        parameter="attributeDistancesUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity-distances",
                        content_type=["application/zip"],
                        required=True,
                    )
                ],
            ),
            tags=Aggregator.instance.tags,
        )


@AGGREGATOR_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Aggregator plugin."""

    @AGGREGATOR_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the Aggregator plugin.",
    )
    @AGGREGATOR_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @AGGREGATOR_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @AGGREGATOR_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the Aggregator plugin.",
    )
    @AGGREGATOR_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @AGGREGATOR_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=Aggregator.instance.name,
                version=Aggregator.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{AGGREGATOR_BLP.name}.CalcSimilarityView"),
            )
        )


@AGGREGATOR_BLP.route("/process/")
class CalcSimilarityView(MethodView):
    """Start a long running processing task."""

    @AGGREGATOR_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @AGGREGATOR_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @AGGREGATOR_BLP.require_jwt("jwt", optional=True)
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


class Aggregator(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Aggregates attribute distances to entity distances."
    tags = ["aggregator"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return AGGREGATOR_BLP

    def get_requirements(self) -> str:
        return ""


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Aggregator.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new aggregation calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    attribute_distances_url = input_params.attribute_distances_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: attribute_distances_url='{attribute_distances_url}'"
    )
    aggregator = input_params.aggregator
    TASK_LOGGER.info(f"Loaded input parameters from db: aggregator='{aggregator}'")

    # load data from file

    attribute_distances = {}

    for file, file_name in get_files_from_zip_url(attribute_distances_url):
        # removes .json from file name to get the name of the attribute
        attr_name = file_name[:-5]

        attribute_distances[attr_name] = json.load(file)

    entity_distance_lists = {}

    for attr_dist in attribute_distances.values():
        for ent in attr_dist:
            ent_ids = (ent["entity_1_ID"], ent["entity_2_ID"])

            if ent_ids in entity_distance_lists:
                entity_distance_lists[ent_ids].append(ent)
            else:
                entity_distance_lists[ent_ids] = [ent]

    entity_distances = []

    for ent_dist_list in entity_distance_lists.values():
        ent_dist = 0.0
        dist_list = [ent["distance"] for ent in ent_dist_list]

        if aggregator == AggregatorsEnum.mean:
            for dist in dist_list:
                ent_dist += dist

            ent_dist /= len(ent_dist_list)
        elif aggregator == AggregatorsEnum.median:
            dist_list = sorted(dist_list)

            if len(dist_list) % 2 == 0:
                ent_dist = (
                    0.5 * dist_list[len(dist_list) // 2]
                    + 0.5 * dist_list[len(dist_list) // 2 - 1]
                )
            else:
                ent_dist = dist_list[len(dist_list) // 2]
        elif aggregator == AggregatorsEnum.max:
            ent_dist = max(dist_list)
        elif aggregator == AggregatorsEnum.min:
            ent_dist = min(dist_list)
        else:
            raise ValueError("Unknown aggregator")

        ent_1_id = ent_dist_list[0]["entity_1_ID"]
        ent_2_id = ent_dist_list[0]["entity_2_ID"]

        entity_distances.append(
            {
                "ID": ent_1_id + "_" + ent_2_id,
                "entity_1_ID": ent_1_id,
                "entity_2_ID": ent_2_id,
                "href": "",
                "distance": ent_dist,
            }
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entity_distances, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "entity_distances.json",
            "entity-distances",
            "application/json",
        )

    return "Result stored in file"
