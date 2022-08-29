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
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional

import flask
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
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "visualization"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


VIS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Visualization plugin API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        entity_points_url: str,
        clusters_url: str,
    ):
        self.entity_points_url = entity_points_url
        self.clusters_url = clusters_url


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
    clusters_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="clusters",
        data_content_types="application/json",
        metadata={
            "label": "Clusters URL",
            "description": "URL to a json file with the clusters.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@VIS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @VIS_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @VIS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Visualization endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Visualization",
            description=VIS.instance.description,
            name=VIS.instance.name,
            version=VIS.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{VIS_BLP.name}.CalcView"),
                ui_href=url_for(f"{VIS_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity-points",
                        content_type=["application/json"],
                        required=True,
                        parameter="entityPointsUrl",
                    ),
                    InputDataMetadata(
                        data_type="clusters",
                        content_type=["application/json"],
                        required=True,
                        parameter="clustersUrl",
                    ),
                ],
                data_output=[],
            ),
            tags=VIS.instance.tags,
        )


@VIS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Visualization plugin."""

    @VIS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the Visualization plugin.",
    )
    @VIS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @VIS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @VIS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the Visualization plugin.",
    )
    @VIS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @VIS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)
        fields = InputParametersSchema().fields
        app = flask.current_app

        # define default values
        default_values = {}

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=VIS.instance.name,
                version=VIS.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{VIS_BLP.name}.CalcView"),
            )
        )


@VIS_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @VIS_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @VIS_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @VIS_BLP.require_jwt("jwt", optional=True)
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


class VIS(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Plots points with cluster information."
    tags = ["visualization"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return VIS_BLP

    def get_requirements(self) -> str:
        return "plotly~=5.3.1\npandas~=1.4.2"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{VIS.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    import pandas as pd
    import plotly.express as px

    # get parameters

    TASK_LOGGER.info(f"Starting new MDS calculation task with db id '{db_id}'")
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
    clusters_url = input_params.clusters_url
    TASK_LOGGER.info(f"Loaded input parameters from db: clusters_url='{clusters_url}'")

    # load data from file

    entity_points = open_url(entity_points_url).json()
    clusters_entities = open_url(clusters_url).json()

    clusters = {}
    ids = []
    points_x = []
    points_y = []
    cluster_list = []
    colors = []

    for ent in clusters_entities:
        clusters[ent["ID"]] = ent["cluster"]

    for ent in entity_points:
        ids.append(ent["ID"])
        points_x.append(ent["point"][0])
        points_y.append(ent["point"][1])

        cluster = clusters[ent["ID"]]
        cluster_list.append(str(cluster))

        if cluster == 0:
            colors.append("red")
        elif cluster == 1:
            colors.append("blue")
        else:
            raise ValueError("Too many clusters.")

    df = pd.DataFrame(
        {
            "x": points_x,
            "y": points_y,
            "ID": ids,
            "cluster": cluster_list,
            "size": [10 for _ in range(len(ids))],
        }
    )

    fig = px.scatter(
        df, x="x", y="y", hover_name="ID", color="cluster", symbol="cluster", size="size"
    )

    with SpooledTemporaryFile(mode="wt") as output:
        html = fig.to_html()
        output.write(html)

        STORE.persist_task_result(
            db_id,
            output,
            "plot.html",
            "plot",
            "text/html",
        )

    return "Result stored in file"
