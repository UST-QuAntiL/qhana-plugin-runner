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
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional

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

from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
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
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "pca"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


PCA_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="PCA plugin API",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class SolverEnum(Enum):
    auto = "auto"
    full = "full"
    arpack = "arpack"
    randomized = "randomized"


class InputParameters:
    def __init__(
        self, entity_points_url: str, dimensions: int, solver: SolverEnum, scale=False
    ):
        self.entity_points_url = entity_points_url
        self.dimensions = dimensions
        self.solver = solver
        self.scale = scale


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Entity points URL",
            "description": "URL to a json file with the entity points.",
            "input_type": "text",
        },
    )
    dimensions = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Dimensions ('auto', if d <= 0)",
            "description": "Number of dimensions the output will have.",
            "input_type": "text",
        },
    )
    solver = EnumField(
        SolverEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Solver",
            "description": "Type of PCA solver that will be used.",
            "input_type": "select",
        },
    )
    scale = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Scale features",
            "description": "Tells, if features should be scaled to be between 0 and 1 or not",
            "input_type": "checkbox",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@PCA_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PCA_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @PCA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """PCA endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Principle Component Analysis (PCA)",
            description=PCA.instance.description,
            name=PCA.instance.name,
            version=PCA.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{PCA_BLP.name}.ProcessView"),
                ui_href=url_for(f"{PCA_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity-points",
                        content_type=["application/json"],
                        required=True,
                        parameter="entityPointsUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="principle-components",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
            ),
            tags=PCA.instance.tags,
        )


@PCA_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the PCA plugin."""

    @PCA_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the PCA plugin.")
    @PCA_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @PCA_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @PCA_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the PCA plugin.")
    @PCA_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @PCA_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)
        fields = InputParametersSchema().fields

        # define default values
        default_values = {
            fields["dimensions"].data_key: 0,
            fields["solver"].data_key: SolverEnum.auto,
            fields["scale"].data_key: False,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=PCA.instance.name,
                version=PCA.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{PCA_BLP.name}.ProcessView"),
            )
        )


@PCA_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @PCA_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @PCA_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @PCA_BLP.require_jwt("jwt", optional=True)
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


class PCA(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "Reduces number of dimensions. (New ONB are the d first principle components)"
    )
    tags = ["dimension-reduction"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return PCA_BLP

    def get_requirements(self) -> str:
        return "scikit-learn~=1.1"


TASK_LOGGER = get_task_logger(__name__)


def load_from_url(entity_point_url):
    import numpy as np

    entity_points = open_url(entity_point_url).json()
    id_to_idx = {}

    idx = 0

    for ent in entity_points:
        if ent["ID"] in id_to_idx:
            raise ValueError("Duplicate ID: ", ent["ID"])

        id_to_idx[ent["ID"]] = idx
        idx += 1

    points_cnt = len(id_to_idx)
    dimensions = len(entity_points[0]["point"])
    points_arr = np.zeros((points_cnt, dimensions))

    for ent in entity_points:
        idx = id_to_idx[ent["ID"]]
        points_arr[idx] = ent["point"]

    return points_arr, id_to_idx


@CELERY.task(name=f"{PCA.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    from sklearn.decomposition import PCA

    # get parameters
    TASK_LOGGER.info(f"Starting new PCA calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entity_points_url = input_params.entity_points_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entity_points_url='{entity_points_url}'"
    )
    dimensions = input_params.dimensions
    TASK_LOGGER.info(f"Loaded input parameters from db: dimensions='{dimensions}'")
    solver = input_params.solver.value
    TASK_LOGGER.info(f"Loaded input parameters from db: solver='{solver}'")
    scale = input_params.scale
    TASK_LOGGER.info(f"Loaded input parameters from db: scale='{scale}'")

    # load data from file
    (entity_points, id_to_idx) = load_from_url(entity_points_url)

    # execute PCA
    if dimensions <= 0:
        dimensions = "mle"
    pca = PCA(n_components=dimensions, svd_solver=solver)
    scaling_factors = []
    if scale:
        entity_points_scaled = entity_points.copy()
        for i in range(entity_points_scaled.shape[1]):
            scaling_factors.append(1.0 / entity_points_scaled[:, i].max())
            entity_points_scaled[:, i] = (
                entity_points_scaled[:, i] / entity_points_scaled[:, i].max()
            )
        pca.fit(entity_points_scaled)
        transformed_points = pca.transform(entity_points)
    else:
        transformed_points = pca.fit_transform(entity_points)

    # prepare output
    entity_points = []
    for ID, i in id_to_idx.items():
        entity_points.append({"ID": ID, "href": "", "point": list(transformed_points[i])})
    if scale:
        entity_points.append(
            {
                "pcaComponents": pca.components_.tolist(),
                "pcaMean": pca.mean_.tolist(),
                "pcaScalingFactors": scaling_factors,
            }
        )
    else:
        entity_points.append(
            {"pcaComponents": pca.components_.tolist(), "pcaMean": pca.mean_.tolist()}
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entity_points, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "pca.json",
            "principle-components",
            "application/json",
        )

    return "Result stored in file"
