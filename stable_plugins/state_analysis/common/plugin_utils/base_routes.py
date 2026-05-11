# Copyright 2026 QHAna plugin runner contributors.
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
from http import HTTPStatus

from flask import Response, abort, redirect, render_template, request, url_for
from flask.views import MethodView
from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result


class BaseMicroFrontend(MethodView):

    PLUGIN = None
    SCHEMA_CLASS = None
    HELP_TEXT = ""
    EXAMPLE_INPUTS = {}

    def render_view(self, data, errors, valid):
        plugin = self.PLUGIN.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = self.SCHEMA_CLASS()
        # TODO: rework result rendering — microfrontend should follow the redirect
        # from ProcessView to TaskView and not read ``task.result`` directly.
        result = None
        task_id = data.get("task_id")
        if task_id:
            task = ProcessingTask.get_by_id(task_id)
            if task:
                result = task.result
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                result=result,
                process=url_for(f"{plugin.identifier}.ProcessView"),
                help_text=self.HELP_TEXT,
                example_values=url_for(
                    f"{plugin.identifier}.MicroFrontend", **self.EXAMPLE_INPUTS
                ),
            )
        )

    def get(self, errors):
        return self.render_view(request.args, errors, valid=False)

    def post(self, errors):
        return self.render_view(request.form, errors, valid=(not errors))


class BaseProcessView(MethodView):
    task_function = None

    def post(self, arguments):
        db_task = ProcessingTask(
            task_name=self.task_function.name, parameters=json.dumps(arguments)
        )
        db_task.save(commit=True)
        task_chain = self.task_function.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        task_chain.link_error(save_task_error.s(db_id=db_task.id))
        task_chain.apply_async()
        db_task.save(commit=True)
        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)),
            code=HTTPStatus.SEE_OTHER,
        )


class BasePluginView(MethodView):
    """Generic plugin-metadata view.

    Subclasses must set ``PLUGIN`` and may override ``DATA_OUTPUT`` and
    ``DATA_INPUT`` with plugin-specific entries.

    TODO: audit ``plugin.identifier`` of every plugin for URL safety. Names
    are unescaped into ``url_for`` here and a non-URL-safe identifier
    would silently produce broken links."""

    PLUGIN = None  # must be set in the subclass
    DATA_INPUT = []  # plugin-specific input metadata
    DATA_OUTPUT = []  # plugin-specific output metadata

    def get(self):
        plugin = self.PLUGIN.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{plugin.identifier}.ProcessView"),
                ui_href=url_for(f"{plugin.identifier}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=self.DATA_INPUT,
                data_output=self.DATA_OUTPUT,
            ),
            tags=plugin.tags,
        )
