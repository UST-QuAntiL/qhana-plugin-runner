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
import json
from enum import Enum
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, Dict

import marshmallow as ma
import numpy as np
import requests
from celery import chain
from celery.utils.log import get_task_logger
from flask import Response, abort, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from scipy.optimize import minimize

from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    InteractionEndpoint,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "optimizer-demo"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


OPTIMIZER_DEMO_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="API of an optimizer plugin that can be used by other plugins.",
    template_folder="hello_world_templates",
)


class Optimizers(Enum):
    cobyla = "COBYLA"
    # nelder_mead = "Nelder-Mead"


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class HyperparametersSchema(FrontendFormBaseSchema):
    optimizer = EnumField(
        Optimizers,
        required=True,
        allow_none=False,
        metadata={
            "label": "Optimizer.",
            "description": "Which optimizer to use.",
            "input_type": "select",
        },
    )
    callback_url = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={
            "label": "Callback URL",
            "description": "Callback URL of the optimizer plugin. Will be filled automatically when using the optimizer plugin. MUST NOT BE CHANGED!",
            "input_type": "text",
        },
    )


class OptimizationInputSchema(MaBaseSchema):
    dataset = ma.fields.Url(required=True, allow_none=False)
    optimizer_db_id = ma.fields.Integer(required=True, allow_none=False)
    number_of_parameters = ma.fields.Integer(required=True, allow_none=False)
    obj_func_db_id = ma.fields.Integer(required=True, allow_none=False)
    obj_func_calc_url = ma.fields.Url(required=True, allow_none=False)


class OptimizationOutputSchema(MaBaseSchema):
    last_objective_value = ma.fields.Float(required=True, allow_none=False)
    optimized_parameters = ma.fields.List(
        ma.fields.Float(), required=True, allow_none=False
    )


@OPTIMIZER_DEMO_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @OPTIMIZER_DEMO_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=OptimizerDemo.instance.name,
            description=OPTIMIZER_DEMO_BLP.description,
            name=OptimizerDemo.instance.identifier,
            version=OptimizerDemo.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{OPTIMIZER_DEMO_BLP.name}.Setup"),
                ui_href=url_for(f"{OPTIMIZER_DEMO_BLP.name}.MicroFrontend"),
                interaction_endpoints=[
                    InteractionEndpoint(
                        type="start-optimization",
                        href=url_for(f"{OPTIMIZER_DEMO_BLP.name}.Optimization"),
                    )
                ],
                plugin_dependencies=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )
                ],
            ),
            tags=["optimizer"],
        )


@OPTIMIZER_DEMO_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend of the optimizer plugin."""

    @OPTIMIZER_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the optimizer plugin."
    )
    @OPTIMIZER_DEMO_BLP.arguments(
        HyperparametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @OPTIMIZER_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the optimizer plugin."
    )
    @OPTIMIZER_DEMO_BLP.arguments(
        HyperparametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        plugin = OptimizerDemo.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparametersSchema()

        return Response(
            render_template(
                "simple_template.html",
                name=OptimizerDemo.instance.name,
                version=OptimizerDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTIMIZER_DEMO_BLP.name}.Setup"
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OPTIMIZER_DEMO_BLP.name}.MicroFrontend"
                ),  # URL of this endpoint
            )
        )


@OPTIMIZER_DEMO_BLP.route("/setup/")
class Setup(MethodView):
    """Start the setup task."""

    @OPTIMIZER_DEMO_BLP.arguments(HyperparametersSchema(unknown=EXCLUDE), location="form")
    @OPTIMIZER_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the setup task."""
        schema = HyperparametersSchema()

        db_task = ProcessingTask(
            task_name=setup_task.name, parameters=schema.dumps(arguments)
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = setup_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)),
            HTTPStatus.SEE_OTHER,
        )


class OptimizerDemo(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return OPTIMIZER_DEMO_BLP

    def get_requirements(self) -> str:
        return "scipy~=1.8.1"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{OptimizerDemo.instance.identifier}.setup_task", bind=True)
def setup_task(self, db_id: int) -> str:
    """
    Retrieves the input data from the database and stores metadata and hyperparameters into files.

    :param self:
    :param db_id: database ID that will be used to retrieve the task data from the database
    :return: log message
    """
    TASK_LOGGER.info(f"Starting setup task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    parameters: Dict = json.loads(task_data.parameters)
    optimizer: str = parameters.get("optimizer")
    callback_url: str = parameters.get("callbackUrl")

    TASK_LOGGER.info(f"Loaded data from db: optimizer='{optimizer}'")
    TASK_LOGGER.info(f"Loaded data from db: callback_url='{callback_url}'")

    if optimizer is None or callback_url is None:
        raise ValueError("Input parameters incomplete")

    with SpooledTemporaryFile(mode="w") as output:
        output.write(task_data.parameters)
        STORE.persist_task_result(
            db_id,
            output,
            "hyperparameters.json",
            "objective-function-hyperparameters",
            "application/json",
        )

    requests.post(
        callback_url,
        json={"dbId": db_id},
    )

    return "Stored metadata and hyperparameters"


@OPTIMIZER_DEMO_BLP.route("/optimization/")
class Optimization(MethodView):
    """Start the optimization."""

    @OPTIMIZER_DEMO_BLP.arguments(OptimizationInputSchema(unknown=EXCLUDE))
    @OPTIMIZER_DEMO_BLP.response(HTTPStatus.OK, OptimizationOutputSchema())
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the optimization."""
        dataset_url: str = arguments["dataset"]
        optimizer_db_id: int = arguments["optimizer_db_id"]
        number_of_parameters: int = arguments["number_of_parameters"]
        obj_func_db_id: int = arguments["obj_func_db_id"]
        obj_func_calc_url: str = arguments["obj_func_calc_url"]

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=optimizer_db_id)

        if db_task is None:
            msg = (
                f"Could not load task data with id {optimizer_db_id} to read parameters!"
            )
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        # randomly initialize parameters
        parameters = np.random.normal(size=(number_of_parameters,))

        obj_func = _objective_function_wrapper(
            dataset_url, obj_func_calc_url, obj_func_db_id
        )

        # optimization
        result = minimize(obj_func, parameters, method="COBYLA")

        # get results
        optimized_parameters: np.ndarray = result.x
        last_objective_value = obj_func(optimized_parameters)

        return {
            "last_objective_value": last_objective_value,
            "optimized_parameters": optimized_parameters.tolist(),
        }


def _objective_function_wrapper(
    dataset_url: str, objective_function_calculation_url: str, obj_func_db_id: int
):
    """
    Provides the objective function with additional information that are needed to execute the requests to the objective
    function plugins.

    :param dataset_url:
    :param objective_function_calculation_url:
    :param obj_func_db_id:
    :return: the wrapped objective function that can be used with scipy optimizers
    """

    def objective_function(x: np.ndarray) -> float:
        request_data = {
            "dataSet": dataset_url,
            "dbId": obj_func_db_id,
            "parameters": x.tolist(),
        }
        res = requests.post(
            objective_function_calculation_url,
            json=request_data,
        ).json()

        return res["objectiveValue"]

    return objective_function
