from http import HTTPStatus
from typing import Mapping, Optional

import numpy as np
import requests
from celery import chain
from celery.utils.log import get_task_logger
from flask import Response, abort, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from scipy.optimize import minimize

from . import OPTIMIZER_DEMO_BLP, OptimizerDemo
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    InteractionEndpoint,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from .schemas import (
    OptimizationInputSchema,
    OptimizationOutputSchema,
    HyperparametersSchema,
    TaskResponseSchema,
)
from .tasks import setup_task


TASK_LOGGER = get_task_logger(__name__)


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
                href=url_for(f"{OPTIMIZER_DEMO_BLP.name}.{Setup.__name__}"),
                ui_href=url_for(f"{OPTIMIZER_DEMO_BLP.name}.{MicroFrontend.__name__}"),
                interaction_endpoints=[
                    InteractionEndpoint(
                        type="start-optimization",
                        href=url_for(
                            f"{OPTIMIZER_DEMO_BLP.name}.{Optimization.__name__}"
                        ),
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
                    f"{OPTIMIZER_DEMO_BLP.name}.{Setup.__name__}"
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OPTIMIZER_DEMO_BLP.name}.{MicroFrontend.__name__}"
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
