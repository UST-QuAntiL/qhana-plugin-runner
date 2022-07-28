from http import HTTPStatus
from logging import Logger
from typing import Mapping, Optional

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from .. import OPTI_COORD_BLP, OptimizationCoordinator
from ..schemas import (
    TaskResponseSchema,
    DatasetInputSchema,
    DatasetInput,
    InternalDataSchema,
    InternalData,
)
from ..tasks import start_optimization_task

TASK_LOGGER: Logger = get_task_logger(__name__)


@OPTI_COORD_BLP.route("/<int:db_id>/dataset-selection/")
class DatasetSelectionUI(MethodView):
    """Micro frontend for the selection of the dataset."""

    example_inputs = {}

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the dataset.",
    )
    @OPTI_COORD_BLP.arguments(
        DatasetInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the dataset.",
    )
    @OPTI_COORD_BLP.arguments(
        DatasetInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        schema = DatasetInputSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=OptimizationCoordinator.instance.name,
                version=OptimizationCoordinator.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTI_COORD_BLP.name}.{StartOptimization.__name__}",
                    db_id=db_id,  # URL of the second processing step
                ),
                example_values=url_for(
                    f"{OPTI_COORD_BLP.name}.{DatasetSelectionUI.__name__}",  # URL of the second micro frontend
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@OPTI_COORD_BLP.route("/<int:db_id>/optimize/")
class StartOptimization(MethodView):
    """
    Starts the optimization task.
    """

    @OPTI_COORD_BLP.arguments(DatasetInputSchema(unknown=EXCLUDE), location="form")
    @OPTI_COORD_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: DatasetInput, db_id: int):
        """Start the optimization task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()

        schema = InternalDataSchema()
        internal_data: InternalData = schema.loads(db_task.parameters)

        internal_data.dataset_url = arguments.dataset_url

        db_task.parameters = schema.dumps(internal_data)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = start_optimization_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
