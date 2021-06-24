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
from typing import Optional
from celery.result import AsyncResult
from flask.helpers import url_for

import marshmallow as ma
from flask import Response
from flask.app import Flask
from flask.templating import render_template
from flask.views import MethodView
from celery.utils.log import get_task_logger


from qhana_plugin_runner.api.util import MaBaseSchema, SecurityBlueprint
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "hello-world"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


HELLO_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="Demo plugin API.",
    template_folder="hello_world_templates",
)


class DemoResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@HELLO_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @HELLO_BLP.response(HTTPStatus.OK, DemoResponseSchema())
    @HELLO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Demo endpoint returning the plugin metadata."""
        return {
            "name": HelloWorld.instance.name,
            "version": HelloWorld.instance.version,
            "identifier": HelloWorld.instance.identifier,
        }


@HELLO_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the hello world plugin."""

    @HELLO_BLP.doc(
        responses={
            f"{HTTPStatus.OK}": {
                "description": "Micro frontend of the hello world plugin.",
                "content": {"text/html": {"schema": {"type": "string"}}},
            }
        }
    )
    @HELLO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""
        return Response(
            render_template(
                "hello_template.html",
                name=HelloWorld.instance.name,
                version=HelloWorld.instance.version,
            )
        )


@HELLO_BLP.route("/processor/")
class PluginsView(MethodView):
    """Start a long running processing task."""

    @HELLO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @HELLO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Start the demo task."""
        result: AsyncResult = demo_task.delay("Demo task input.")
        return {
            "name": demo_task.name,
            "task_id": str(result.id),
            "task_result_url": url_for("tasks-api.TaskView", task_id=str(result.id)),
        }


class HelloWorld(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)
        print("\nInitialized hello world plugin.\n")

    def get_api_blueprint(self):
        return HELLO_BLP


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{HelloWorld.instance.identifier}.demo_task")
def demo_task(input_str: str) -> str:
    TASK_LOGGER.info(f"Starting new demo task with input '{input_str}'")
    if input_str:
        return input_str.replace("input", "output")
    return ""
