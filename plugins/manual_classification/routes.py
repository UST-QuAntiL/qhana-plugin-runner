from http import HTTPStatus
from json import dumps, loads
from typing import Mapping, Optional

from celery.canvas import chain
from celery.result import AsyncResult
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
    PluginType,
    EntryPoint,
    DataMetadata,
)

from plugins.manual_classification import MANUAL_CLASSIFICATION_BLP, ManualClassification
from plugins.manual_classification.schemas import (
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

from plugins.manual_classification.tasks import (
    pre_render_classification,
    add_class,
    save_classification,
)


TASK_LOGGER = get_task_logger(__name__)


@MANUAL_CLASSIFICATION_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, ResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Entity filter endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Manual Classification",
            description="Manually annotate classes for data sets from MUSE database.",
            name=ManualClassification.instance.name,
            version=ManualClassification.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.LoadView"),
                ui_href=url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontend"),
                data_input=[  # TODO: only file input (entities...)
                    DataMetadata(
                        data_type="raw",
                        content_type=[
                            "application/json",
                            "application/zip",
                        ],  # TODO: OR -> json, csv... scatch, not finalized yet
                        required=True,
                    )
                ],
                data_output=[  # TODO
                    DataMetadata(
                        data_type="raw", content_type=["application/json"], required=True
                    )
                ],
            ),
            tags=["data-annotation"],
        )


@MANUAL_CLASSIFICATION_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the manual classification plugin."""

    example_inputs = {
        "inputFileUrl": "file:///<path_to_file>/entities.json",
    }

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
        """Start the data preprocessing task."""
        db_task = ProcessingTask(task_name="manual-classification")
        db_task.save(commit=True)

        schema = LoadParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=ManualClassification.instance.name,
                version=ManualClassification.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.LoadView", db_id=db_task.id
                ),
                example_values=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontend",
                    **self.example_inputs,
                ),
            )
        )


@MANUAL_CLASSIFICATION_BLP.route("/<string:db_id>/load/")
class LoadView(MethodView):
    """Start a data preprocessing task."""

    @MANUAL_CLASSIFICATION_BLP.arguments(
        LoadParametersSchema(unknown=EXCLUDE), location="form"
    )
    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: str):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)
        db_task.parameters = dumps(arguments)
        db_task.save(commit=True)

        # add classification step
        step_id = "classification"
        href = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.ClassificationView", db_id=db_task.id
        )
        ui_href = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontendClassification",
            db_id=db_task.id,
        )

        # all tasks need to know about db id to load the db entry
        task: chain = pre_render_classification.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=30
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

        task_data.clear_previous_step(commit=True)

        # retrive data to build frontend
        schema = ClassificationSchema()
        return Response(
            render_template(
                "manual_classification_template.html",
                name=ManualClassification.instance.name,
                version=ManualClassification.instance.version,
                schema=schema,
                values=data,
                id_list=loads(task_data.data["entity_data"]).keys(),
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
        )  # TODO: Only if every form submit is handled as a step clear can the submitted data be correctly recorded in the backend. However this part is still work in progress. It may be a good idea to rely on javascript in the microfrontend to paginate the form for the user client side but submit all inputs as a single bundle. This will be easier to do once the microfrontends are loaded inside of iframes as they will have to run some javascript already (I am currently working on this).


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
        """Start the classification task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = dumps(arguments)
        db_task.save()

        # all tasks need to know about db id to load the db entry
        task: chain = add_class.s(db_id=db_task.id)

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        # TODO: reload classification step, not sure if that's how ui is working...
        return redirect(
            url_for(
                f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontendClassification",
                db_id=db_id,
            )
        )


@MANUAL_CLASSIFICATION_BLP.route("/<string:db_id>/done/")  # TODO
class ClassificationDoneView(MethodView):
    """Start a classification processing task."""

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
