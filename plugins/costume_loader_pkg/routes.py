from http import HTTPStatus
from typing import Mapping

from celery.canvas import chain
from celery.result import AsyncResult
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView

from plugins.costume_loader_pkg import COSTUME_LOADER_BLP, CostumeLoader
from plugins.costume_loader_pkg.schemas import (
    CostumeLoaderUIResponseSchema,
    TaskResponseSchema,
)
from plugins.costume_loader_pkg.tasks import costume_loading_task, taxonomy_loading_task
from qhana_plugin_runner.api.util import FrontendFormBaseSchema
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result


@COSTUME_LOADER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @COSTUME_LOADER_BLP.response(HTTPStatus.OK, CostumeLoaderUIResponseSchema())
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Plugin loader endpoint returning the plugin metadata."""
        return {
            "name": CostumeLoader.instance.name,
            "version": CostumeLoader.instance.version,
            "identifier": CostumeLoader.instance.identifier,
        }


@COSTUME_LOADER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the costume loader plugin."""

    @COSTUME_LOADER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the costume loader plugin."
    )
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""
        return self.render(request.args, {})

    @COSTUME_LOADER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the costume loader plugin."
    )
    @COSTUME_LOADER_BLP.arguments(
        FrontendFormBaseSchema(),
        location="form",
        required=False,
    )
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        return Response(
            render_template(
                "costume_loader_template.html",
                name=CostumeLoader.instance.name,
                version=CostumeLoader.instance.version,
                schema=FrontendFormBaseSchema(),
                values=data,
                errors=errors,
                process=url_for(f"{COSTUME_LOADER_BLP.name}.CostumeLoadingView"),
            )
        )


@COSTUME_LOADER_BLP.route("/load_costumes/")
class CostumeLoadingView(MethodView):
    """Start a long running processing task."""

    @COSTUME_LOADER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def post(self):
        """Start the costume loading task."""
        db_task = ProcessingTask(
            task_name=costume_loading_task.name,
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = costume_loading_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(result.id)), HTTPStatus.SEE_OTHER
        )


@COSTUME_LOADER_BLP.route("/load_taxonomy/")
class TaxonomyLoadingView(MethodView):
    """Start a long running processing task."""

    @COSTUME_LOADER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def post(self):
        """Start the taxonomy loading task."""
        db_task = ProcessingTask(task_name=taxonomy_loading_task.name)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = taxonomy_loading_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(result.id)), HTTPStatus.SEE_OTHER
        )
