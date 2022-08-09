# Support Vector Machines (classical and quantum)
from http import HTTPStatus
from typing import Optional, Mapping
from json import dumps

import marshmallow as ma
from marshmallow import EXCLUDE
from flask import abort, redirect
from flask.app import Flask
from flask.views import MethodView
from flask.globals import request
from flask.helpers import url_for
from flask.wrappers import Response
from flask.templating import render_template
from celery.canvas import chain
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    EntryPoint,
)

from qhana_plugin_runner.api.util import (
    SecurityBlueprint,
    MaBaseSchema,
    FrontendFormBaseSchema,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier


_plugin_name = "SVM"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

SVM_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Support Vector Machine API",
    # template_folder="hello_world_templates",
)


# class DemoResponseSchema(MaBaseSchema):
#     name = ma.fields.String(required=True, allow_none=False, dump_only=True)
#     version = ma.fields.String(required=True, allow_none=False, dump_only=True)
#     identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class SVMSchema(FrontendFormBaseSchema):
    print("Schema")
    # INPUT....
    # print("TODO")  # TODO
    input_str = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Input String",
            "description": "A simple string input.",
            "input_type": "textarea",
        },
    )


@SVM_BLP.route("/")
class PluginView(MethodView):
    """Plugins collection resource."""

    print("/")

    @SVM_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @SVM_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = SVM.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=SVM_BLP.description,
            name=plugin.identifier,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{SVM_BLP.name}.ProcessView"),
                ui_href=url_for(f"{SVM_BLP.name}.MicroFrontend"),
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
            tags=[],
        )


@SVM_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the SVM plugin."""

    print("/ui/")

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @SVM_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the SVM plugin.")
    @SVM_BLP.arguments(
        SVMSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @SVM_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        print("POST")
        return self.render(request.args, errors)

    def render(self, data: Mapping, errors: dict):
        print("RENDER")
        plugin = SVM.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = SVMSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{SVM_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{SVM_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@SVM_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    print("/process/")

    @SVM_BLP.arguments(SVMSchema(unknown=EXCLUDE), location="form")
    @SVM_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @SVM_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(task_name=demo_task.name, parameters=dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = demo_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class SVM(QHAnaPluginBase):
    print("SVM")
    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return SVM_BLP


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{SVM.instance.identifier}.demo_task", bind=True)
def demo_task(self, db_id: int) -> str:
    print("TASK")
    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    return "DONE"


# TODO
# when clicking on SVM plugin in plugin runner:
# {"code":405, "status":"Method Not Allowed"}
