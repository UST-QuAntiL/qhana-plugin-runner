from http import HTTPStatus
from json import dumps
from typing import Mapping, Optional

from celery.canvas import chain
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from celery.utils.log import get_task_logger
from marshmallow.utils import INCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    EntryPoint,
    DataMetadata,
    InputDataMetadata,
)

from . import MANUAL_CLASSIFICATION_BLP, ManualClassification
from .schemas import (
    ResponseSchema,
    TaskResponseSchema,
    LoadParametersSchema,
    ClassificationSchema,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import (
    add_step,
    save_task_error,
    save_task_result,
)

from .tasks import pre_render_classification, add_class, save_classification


TASK_LOGGER = get_task_logger(__name__)


@MANUAL_CLASSIFICATION_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Manual classification endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Manual Classification",
            description=ManualClassification.instance.description,
            name=ManualClassification.instance.name,
            version=ManualClassification.instance.version,
            type=PluginType.complex,
            entry_point=EntryPoint(
                href=url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.LoadView"),
                ui_href=url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/list",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="inputFileUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/list",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                    ),
                ],
            ),
            tags=ManualClassification.instance.tags,
        )


@MANUAL_CLASSIFICATION_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the manual classification plugin."""

    example_inputs = {}

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the manual classification plugin."
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        LoadParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the manual classification plugin."
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        LoadParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        return Response(
            render_template(
                "simple_template.html",
                name=ManualClassification.instance.name,
                version=ManualClassification.instance.version,
                schema=LoadParametersSchema(),
                values=data,
                errors=errors,
                process=url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.LoadView"),
                example_values=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontend",
                    **self.example_inputs,
                ),
            )
        )


@MANUAL_CLASSIFICATION_BLP.route("/load/")
class LoadView(MethodView):
    """Start a data preprocessing task."""

    @MANUAL_CLASSIFICATION_BLP.arguments(
        LoadParametersSchema(unknown=EXCLUDE), location="form"
    )
    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        db_task = ProcessingTask(task_name="manual-classification")
        db_task.parameters = dumps(arguments)
        db_task.save(commit=True)

        # add classification step
        step_id = "annotate-class-1"
        href = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.ClassificationView",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontendClassification",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["step_id"] = 1

        # all tasks need to know about db id to load the db entry
        task: chain = pre_render_classification.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=(db_task.data["step_id"] / (db_task.data["step_id"] + 1)) * 100,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@MANUAL_CLASSIFICATION_BLP.route("/<string:db_id>/classification-ui/")
class MicroFrontendClassification(MethodView):
    """Micro frontend for the classification step of the manual classification plugin."""

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the classification step of the manual classification plugin.",
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        ClassificationSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: str):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id)

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the classification step of the manual classification plugin.",
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        ClassificationSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: str):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id)

    def render(self, data: Mapping, errors: dict, db_id: str):
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if task_data is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        # retrive data to build frontend
        schema = ClassificationSchema()
        return Response(
            render_template(
                "manual_classification_template.html",
                name=ManualClassification.instance.name,
                version=ManualClassification.instance.version,
                schema=schema,
                values=data,
                entity_list=task_data.data["entity_list"],
                attr_list=task_data.data["attr_list"],
                errors=errors,
                process=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.ClassificationView", db_id=db_id
                ),
                done=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.ClassificationDoneView",
                    db_id=db_id,
                ),
                example_values=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontendClassification",
                    db_id=db_id,
                ),
            )
        )


@MANUAL_CLASSIFICATION_BLP.route("/<string:db_id>/classification/")
class ClassificationView(MethodView):
    """Start a classification processing task."""

    @MANUAL_CLASSIFICATION_BLP.arguments(
        ClassificationSchema(unknown=INCLUDE),
        location="form",  # TODO: this should cause fields not in schema (id's) to be included... not sure if this works
    )
    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: str):
        """Start the classification task and add another step."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = dumps(arguments)
        db_task.clear_previous_step(commit=True)
        db_task.save()

        # add classification step
        db_task.data["step_id"] += 1
        step_id = "annotate-class-" + str(db_task.data["step_id"])
        href = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.ClassificationView",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontendClassification",
            db_id=db_task.id,
            _external=True,
        )

        # all tasks need to know about db id to load the db entry
        task: chain = add_class.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=(db_task.data["step_id"] / (db_task.data["step_id"] + 1)) * 100,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


@MANUAL_CLASSIFICATION_BLP.route("/<string:db_id>/done/")
class ClassificationDoneView(MethodView):
    """Start a classification processing task of the last step."""

    @MANUAL_CLASSIFICATION_BLP.arguments(
        ClassificationSchema(unknown=INCLUDE), location="form"
    )
    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: str):
        """Start the classification task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = dumps(arguments)
        db_task.clear_previous_step(commit=True)
        db_task.save()

        # all tasks need to know about db id to load the db entry
        task: chain = (
            add_class.s(db_id=db_id)
            | save_classification.s(db_id=db_id)
            | save_task_result.s(db_id=db_id)
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
