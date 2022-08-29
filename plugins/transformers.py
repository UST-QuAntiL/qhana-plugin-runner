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
import math
from enum import Enum
from http import HTTPStatus
from io import StringIO
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, List
from zipfile import ZipFile

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

_plugin_name = "sim-to-dist-transformers"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


TRANSFORMERS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Similarity to distance transformers plugin API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class TransformersEnum(Enum):
    linear_inverse = "Linear Inverse"
    exponential_inverse = "Exponential Inverse"
    gaussian_inverse = "Gaussian Inverse"
    polynomial_inverse = "Polynomial Inverse"
    square_inverse = "Square Inverse"


class InputParameters:
    def __init__(
        self,
        attribute_similarities_url: str,
        attributes: str,
        transformer: TransformersEnum,
    ):
        self.attribute_similarities_url = attribute_similarities_url
        self.attributes = attributes
        self.transformer = transformer


class InputParametersSchema(FrontendFormBaseSchema):
    attribute_similarities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="attribute-similarities",
        data_content_types="application/zip",
        metadata={
            "label": "Attribute similarities URL",
            "description": "URL to a zip file with the attribute similarities.",
            "input_type": "text",
        },
    )
    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Attributes",
            "description": "Attributes for which the similarity shall be transformed to distance.",
            "input_type": "textarea",
        },
    )
    transformer = EnumField(
        TransformersEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Transformer",
            "description": "Transformer that shall be used to transform the similarities to distances.",
            "input_type": "select",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@TRANSFORMERS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @TRANSFORMERS_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @TRANSFORMERS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Transformers endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Similarities to distances transformers",
            description=Transformers.instance.description,
            name=Transformers.instance.name,
            version=Transformers.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{TRANSFORMERS_BLP.name}.CalcSimilarityView"),
                ui_href=url_for(f"{TRANSFORMERS_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="attribute-similarities",
                        content_type=["application/zip"],
                        required=True,
                        parameter="attributeSimilaritiesUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="attribute-distances",
                        content_type=["application/zip"],
                        required=True,
                    )
                ],
            ),
            tags=Transformers.instance.tags,
        )


@TRANSFORMERS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Sym Max Mean plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @TRANSFORMERS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the similarity to distance transformers plugin.",
    )
    @TRANSFORMERS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @TRANSFORMERS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @TRANSFORMERS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the similarity to distance transformers plugin.",
    )
    @TRANSFORMERS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @TRANSFORMERS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=Transformers.instance.name,
                version=Transformers.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{TRANSFORMERS_BLP.name}.CalcSimilarityView"),
                example_values=url_for(
                    f"{TRANSFORMERS_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@TRANSFORMERS_BLP.route("/process/")
class CalcSimilarityView(MethodView):
    """Start a long running processing task."""

    @TRANSFORMERS_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @TRANSFORMERS_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @TRANSFORMERS_BLP.require_jwt("jwt", optional=True)
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


class Transformers(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Transforms similarities to distances."
    tags = ["sim-to-dist"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return TRANSFORMERS_BLP

    def get_requirements(self) -> str:
        return ""


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Transformers.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new Sym Max Mean calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    attribute_similarities_url = input_params.attribute_similarities_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: attribute_similarities_url='{attribute_similarities_url}'"
    )
    attributes: str = input_params.attributes
    TASK_LOGGER.info(f"Loaded input parameters from db: attributes='{attributes}'")
    attributes: List[str] = attributes.splitlines()
    transformer = input_params.transformer
    TASK_LOGGER.info(f"Loaded input parameters from db: transformer='{transformer}'")

    # load data from file

    element_similarities = {}

    for file, file_name in get_files_from_zip_url(attribute_similarities_url):
        # removes .json from file name to get the name of the attribute
        attr_name = file_name[:-5]

        element_similarities[attr_name] = json.load(file)

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attribute in attributes:
        attribute_distances = []
        attr_elem_sims = element_similarities[attribute]

        for sim_entity in attr_elem_sims:
            sim = sim_entity["similarity"]
            dist = None

            if transformer == TransformersEnum.linear_inverse:
                dist = 1.0 - sim
            elif transformer == TransformersEnum.exponential_inverse:
                dist = math.exp(-sim)
            elif transformer == TransformersEnum.gaussian_inverse:
                dist = math.exp(-sim * sim)
            elif transformer == TransformersEnum.polynomial_inverse:
                alpha = 1.0
                beta = 1.0

                dist = 1.0 / (1.0 + pow(sim / alpha, beta))
            elif transformer == TransformersEnum.square_inverse:
                max_sim = 1.0
                dist = (1.0 / math.sqrt(2.0)) * math.sqrt(2.0 * max_sim - 2 * sim)

            attribute_distances.append(
                {
                    "ID": sim_entity["ID"],
                    "entity_1_ID": sim_entity["entity_1_ID"],
                    "entity_2_ID": sim_entity["entity_2_ID"],
                    "href": "",
                    "distance": dist,
                }
            )

        with StringIO() as file:
            save_entities(attribute_distances, file, "application/json")
            file.seek(0)
            zip_file.writestr(attribute + ".json", file.read())

    zip_file.close()

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        "attr_dist.zip",
        "attribute_distances",
        "application/zip",
    )

    return "Result stored in file"
