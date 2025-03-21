# Copyright 2025 QHAna plugin runner contributors.
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
# limitations under the Licens

from http import HTTPStatus
from json import dumps
from typing import Mapping

from celery.canvas import chain
from flask import abort, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import JOIN_BLP, DataJoin
from .schemas import (
    DataJoinAttrSelectParametersSchema,
    DataJoinBaseParametersSchema,
    DataJoinFinishJoinParametersSchema,
    DataJoinJoinParametersSchema,
)
from .tasks import add_data_to_join, load_base


@JOIN_BLP.route("/")
class PluginRootView(MethodView):
    """Plugin metadata for the data join plugin."""

    @JOIN_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = DataJoin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{JOIN_BLP.name}.{LoadBaseView.__name__}"),
                ui_href=url_for(f"{JOIN_BLP.name}.{MicroFrontendBase.__name__}"),
                plugin_dependencies=[],
                data_input=[
                    InputDataMetadata(
                        data_type="entity/*",
                        content_type=["text/csv", "application/json"],
                        required=True,
                        parameter="base",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/*",
                        content_type=["text/csv", "application/json"],
                        required=True,
                    )
                ],
            ),
            tags=plugin.tags,
        )


@JOIN_BLP.route("/ui/")
class MicroFrontendBase(MethodView):
    """Micro frontend for the data join plugin."""

    @JOIN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the data join plugin."
    )
    @JOIN_BLP.arguments(
        DataJoinBaseParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @JOIN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the data join plugin."
    )
    @JOIN_BLP.arguments(
        DataJoinBaseParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = DataJoin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = DataJoinBaseParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{JOIN_BLP.name}.{LoadBaseView.__name__}"),
                help_text="First select the entities as base you want to join other data to. In a second step select the data you want to join to the base.",
                example_values=url_for(f"{JOIN_BLP.name}.{MicroFrontendBase.__name__}"),
            )
        )


@JOIN_BLP.route("/<int:db_id>/join-ui/")
class MicroFrontendJoin(MethodView):
    """Micro frontend for the data join substep of the data join plugin."""

    @JOIN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the data join plugin."
    )
    @JOIN_BLP.arguments(
        DataJoinJoinParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, False, db_id)

    @JOIN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the data join plugin."
    )
    @JOIN_BLP.arguments(
        DataJoinJoinParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors, db_id)

    def render(self, data: Mapping, errors: dict, valid: bool, db_id: int):
        plugin = DataJoin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        task_data = ProcessingTask.get_by_id(db_id)
        if not task_data or task_data.task_name not in (
            load_base.name,
            add_data_to_join.name,
        ):
            # abort(HTTPStatus.NOT_FOUND)
            pass  # FIXME

        assert isinstance(task_data.data, dict)
        attributes = task_data.data.get("attributes", [])
        assert isinstance(attributes, (list, tuple))

        # FIXME remove
        attributes = ["ID", "href", "name", "year", "reference_id", "something"]

        data_dict = {"join": "", "attribute": "ID"}
        data_dict.update(data)

        if data_dict["attribute"] not in attributes:
            attr_errors = errors.setdefault("attribute", [])
            attr_errors.append(
                f"Attribute '{data_dict['attribute']}' is not present in the join base!"
            )

        joins = task_data.data.get("joins", [])
        joins = [{}]  # FIXME: remove
        if joins:
            done = url_for(f"{JOIN_BLP.name}.{FinishJoinsView.__name__}", db_id=db_id)
        else:
            done = None

        schema = DataJoinJoinParametersSchema()
        return Response(
            render_template(
                "data-join_join.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                schema_extras={"attribute": {"options": {a: a for a in attributes}}},
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{JOIN_BLP.name}.{AddJoinView.__name__}", db_id=db_id),
                done=done,
            )
        )


@JOIN_BLP.route("/<int:db_id>/attribute-selection-ui/")
class MicroFrontendAttributeSelection(MethodView):
    """Micro frontend for the attribute selection substep of the data join plugin."""

    @JOIN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the data join plugin."
    )
    @JOIN_BLP.arguments(
        DataJoinAttrSelectParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, False, db_id)

    @JOIN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the data join plugin."
    )
    @JOIN_BLP.arguments(
        DataJoinAttrSelectParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors, db_id)

    def render(self, data: Mapping, errors: dict, valid: bool, db_id: int):
        plugin = DataJoin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        task_data = ProcessingTask.get_by_id(db_id)
        if not task_data or task_data.task_name not in (
            load_base.name,
            add_data_to_join.name,
        ):
            # abort(HTTPStatus.NOT_FOUND)
            pass  # FIXME

        assert isinstance(task_data.data, dict)
        attributes = task_data.data.get("attributes", [])
        assert isinstance(attributes, (list, tuple))
        base = task_data.data.get("base")
        # assert isinstance(base, dict)
        joins = task_data.data.get("joins", [])
        assert isinstance(joins, (list, tuple))

        # FIXME remove
        attributes = ["ID", "href", "name", "year", "reference_id", "something"]
        base = {"name": "BASE Entities.json"}
        joins = [
            {
                "name": "JOIN Entities.csv",
                "attributes": ["ID", "href", "foo", "bar", "year"],
            }
        ]

        data_dict = {}
        data_dict.update(data)

        schema = DataJoinJoinParametersSchema()
        return Response(
            render_template(
                "data-join_attributes.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(
                    f"{JOIN_BLP.name}.{AttributeSelectionView.__name__}", db_id=db_id
                ),
                base=base,
                base_attrs=attributes,
                joins=joins,
            )
        )


@JOIN_BLP.route("/load-base/")
class LoadBaseView(MethodView):
    """Load the entities that will be the base for the join."""

    @JOIN_BLP.arguments(DataJoinBaseParametersSchema(unknown=EXCLUDE), location="form")
    @JOIN_BLP.response(HTTPStatus.SEE_OTHER)
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Load the entities that will be the base for the join."""
        db_task = ProcessingTask(task_name=load_base.name, parameters=dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = load_base.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@JOIN_BLP.route("/<int:db_id>/join/")
class AddJoinView(MethodView):
    """Add entities that will be joined to the base."""

    @JOIN_BLP.arguments(DataJoinJoinParametersSchema(unknown=EXCLUDE), location="form")
    @JOIN_BLP.response(HTTPStatus.SEE_OTHER)
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Add entities that will be joined to the base."""
        # FIXME implement

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


@JOIN_BLP.route("/<int:db_id>/join/")
class FinishJoinsView(MethodView):
    """Finish selecting joins and proceed to attribute selection."""

    @JOIN_BLP.arguments(
        DataJoinFinishJoinParametersSchema(unknown=EXCLUDE), location="form"
    )
    @JOIN_BLP.response(HTTPStatus.SEE_OTHER)
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Finish selecting joins and proceed to attribute selection."""
        # FIXME implement

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


@JOIN_BLP.route("/<int:db_id>/attribute-selection/")
class AttributeSelectionView(MethodView):
    """Select the attributes to keep for the joined output."""

    @JOIN_BLP.arguments(
        DataJoinAttrSelectParametersSchema(unknown=EXCLUDE), location="form"
    )
    @JOIN_BLP.response(HTTPStatus.SEE_OTHER)
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Select the attributes to keep for the joined output."""
        # FIXME implement
        print("\n\n", arguments, "\n\n")

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
