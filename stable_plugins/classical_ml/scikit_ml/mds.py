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

import flask
import marshmallow as ma
from celery.canvas import chain
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
    SecurityBlueprint,
    FileUrl,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.requests import open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "mds"
__version__ = "v0.2.0"
_identifier = plugin_identifier(_plugin_name, __version__)


MDS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="MDS plugin API.",
)


class MetricEnum(Enum):
    metric_mds = "Metric MDS"
    nonmetric_mds = "Nonmetric MDS"


class InputParameters:
    def __init__(
        self,
        entity_distances_url: str,
        dimensions: int,
        metric: MetricEnum,
        n_init: int,
        max_iter: int,
    ):
        self.entity_distances_url = entity_distances_url
        self.dimensions = dimensions
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter


class InputParametersSchema(FrontendFormBaseSchema):
    entity_distances_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="custom/entity-distances",
        data_content_types="application/json",
        metadata={
            "label": "Entity distances URL",
            "description": "URL to a json file with the entity distances.",
            "input_type": "text",
        },
    )
    dimensions = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Dimensions",
            "description": "Number of dimensions the output will have.",
            "input_type": "text",
        },
    )
    metric = EnumField(
        MetricEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Metric",
            "description": "Type of MDS that will be used.",
            "input_type": "select",
        },
    )
    n_init = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "SMACOF executions",
            "description": "Number of times SMACOF will be executed with different initial values.",
            "input_type": "text",
        },
    )
    max_iter = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "SMACOF max iterations",
            "description": "Maximum number of SMACOF iterations.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@MDS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @MDS_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @MDS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """MDS endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Multidimensional Scaling (MDS)",
            description=MDS.instance.description,
            name=MDS.instance.name,
            version=MDS.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{MDS_BLP.name}.CalcView"),
                ui_href=url_for(f"{MDS_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="custom/entity-distances",
                        content_type=["application/json"],
                        required=True,
                        parameter="entityDistancesUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
            ),
            tags=MDS.instance.tags,
        )


@MDS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the MDS plugin."""

    @MDS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the MDS plugin.",
    )
    @MDS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MDS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @MDS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the MDS plugin.",
    )
    @MDS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MDS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        data_dict = dict(data)
        fields = InputParametersSchema().fields
        app = flask.current_app

        # define default values
        default_values = {
            fields["dimensions"].data_key: 2,
            fields["metric"].data_key: MetricEnum.metric_mds,
            fields["n_init"].data_key: 4,
            fields["max_iter"].data_key: 300,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=MDS.instance.name,
                version=MDS.instance.version,
                schema=InputParametersSchema(),
                valid=valid,
                values=data_dict,
                errors=errors,
                process=url_for(f"{MDS_BLP.name}.CalcView"),
            )
        )


@MDS_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @MDS_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @MDS_BLP.response(HTTPStatus.SEE_OTHER)
    @MDS_BLP.require_jwt("jwt", optional=True)
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


class MDS(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Converts distance values (distance matrix) to points in a space."
    tags = ["dist-to-points"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return MDS_BLP

    def get_requirements(self) -> str:
        return "scikit-learn~=1.1"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{MDS.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    import numpy as np
    from sklearn import manifold

    # get parameters

    TASK_LOGGER.info(f"Starting new MDS calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entity_distances_url = input_params.entity_distances_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entity_distances_url='{entity_distances_url}'"
    )
    dimensions = input_params.dimensions
    TASK_LOGGER.info(f"Loaded input parameters from db: dimensions='{dimensions}'")
    metric = input_params.metric
    TASK_LOGGER.info(f"Loaded input parameters from db: metric='{metric}'")
    n_init = input_params.n_init
    TASK_LOGGER.info(f"Loaded input parameters from db: n_init='{n_init}'")
    max_iter = input_params.max_iter
    TASK_LOGGER.info(f"Loaded input parameters from db: max_iter='{max_iter}'")

    # load data from file

    entity_distances = open_url(entity_distances_url).json()
    id_to_idx = {}

    idx = 0

    for ent_dist in entity_distances:
        if ent_dist["entity_1_ID"] not in id_to_idx:
            id_to_idx[ent_dist["entity_1_ID"]] = idx
            idx += 1

    distance_matrix = np.zeros((len(id_to_idx), len(id_to_idx)))

    for ent_dist in entity_distances:
        ent_1_id = ent_dist["entity_1_ID"]
        ent_2_id = ent_dist["entity_2_ID"]
        dist = ent_dist["distance"]

        ent_1_idx = id_to_idx[ent_1_id]
        ent_2_idx = id_to_idx[ent_2_id]

        distance_matrix[ent_1_idx, ent_2_idx] = dist
        distance_matrix[ent_2_idx, ent_1_idx] = dist

    mds = manifold.MDS(
        dimensions,
        metric=metric == MetricEnum.metric_mds,
        n_init=n_init,
        max_iter=max_iter,
        dissimilarity="precomputed",
    )

    transformed = mds.fit_transform(distance_matrix)

    entity_points = []
    dim_attributes = _get_dim_attributes(dimensions)

    for ent_id, idx in id_to_idx.items():
        new_entity_point = {"ID": ent_id, "href": ""}
        new_entity_point.update({d: x for d, x in zip(dim_attributes, transformed[idx])})

        entity_points.append(new_entity_point)

    metric_name = str(metric.name).removesuffix("_mds")
    filename = retrieve_filename(entity_distances_url)
    info_str = f"_mds_dim_{dimensions}_metric_{metric_name}_from_{filename}"

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entity_points, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"entity_points{info_str}.json",
            "entity/vector",
            "application/json",
        )

    return "Result stored in file"


def _get_dim_attributes(dim):
    """
    Returns the attributes for each dimension, with the correct length.
    For each dimension we have an attribute, i.e. dimension 0 = dim0, dimension 1 = dim1, ...
    This method adds a zero padding to ensure that every dim<int> has the same length, e.g. dim00, dim01, ..., dim10, dim11
    :params dim: int number of dimensions
    :return: list[str]
    """
    zero_padding = len(str(dim - 1))
    dim_attributes = [f"dim{d:0{zero_padding}}" for d in range(dim)]
    return dim_attributes
