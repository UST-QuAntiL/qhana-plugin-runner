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

from celery.canvas import Signature, chain
from celery.utils.log import get_task_logger
from flask import abort, redirect
from flask.globals import current_app, request
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
from qhana_plugin_runner.tasks import (
    TASK_STEPS_CHANGED,
    add_step,
    save_task_error,
    save_task_result,
)

from . import JOIN_BLP, DataJoin
from .schemas import (
    DataJoinAttrSelectParametersSchema,
    DataJoinBaseParametersSchema,
    DataJoinFinishJoinParametersSchema,
    DataJoinJoinParametersSchema,
)
from .tasks import add_data_to_join, join_data, load_base

TASK_LOGGER = get_task_logger(__name__)


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
                ui_href=url_for(f"{JOIN_BLP.name}.{LoadBaseMicroFrontend.__name__}"),
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
class LoadBaseMicroFrontend(MethodView):
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
                example_values=url_for(
                    f"{JOIN_BLP.name}.{LoadBaseMicroFrontend.__name__}"
                ),
            )
        )


@JOIN_BLP.route("/<int:db_id>/<step_id>/add-join-ui/")
class AddJoinMicroFrontend(MethodView):
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
    def get(self, errors, db_id: int, step_id: str):
        """Return the micro frontend."""
        return self.render(request.args, errors, False, db_id, step_id)

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
    def post(self, errors, db_id: int, step_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors, db_id, step_id)

    def render(self, data: Mapping, errors: dict, valid: bool, db_id: int, step_id: int):
        plugin = DataJoin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        task_data = ProcessingTask.get_by_id(db_id)
        if not task_data or task_data.task_name not in (
            load_base.name,
            add_data_to_join.name,
        ):
            abort(HTTPStatus.NOT_FOUND)

        assert isinstance(task_data.data, dict)
        attributes = task_data.data.get("attributes", [])
        assert isinstance(attributes, (list, tuple))

        data_dict = {"join": "", "attribute": attributes[0]}
        data_dict.update(data)

        if data_dict["attribute"] not in attributes:
            attr_errors = errors.setdefault("attribute", [])
            attr_errors.append(
                f"Attribute '{data_dict['attribute']}' is not present in the join base!"
            )

        joins = task_data.data.get("joins", [])
        if joins:
            done = url_for(
                f"{JOIN_BLP.name}.{FinishJoinsView.__name__}",
                db_id=db_id,
            )
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
                process=url_for(
                    f"{JOIN_BLP.name}.{AddJoinView.__name__}",
                    db_id=db_id,
                    step_id=step_id,
                ),
                done=done,
            )
        )


@JOIN_BLP.route("/<int:db_id>/attribute-selection-ui/")
class AttributeSelectionMicroFrontend(MethodView):
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
            abort(HTTPStatus.NOT_FOUND)

        assert isinstance(task_data.data, dict)
        attributes = task_data.data.get("attributes", [])
        assert isinstance(attributes, (list, tuple))
        base = task_data.data.get("base")
        assert isinstance(base, dict)
        joins = task_data.data.get("joins", [])
        assert isinstance(joins, (list, tuple))

        if not joins:
            abort(
                HTTPStatus.NOT_FOUND,
                "Task has not yet reached the attribute selection step!",
            )

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

        # next step
        step_id = "add-join"
        href = url_for(
            f"{JOIN_BLP.name}.{AddJoinView.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{JOIN_BLP.name}.{AddJoinMicroFrontend.__name__}",
            db_id=db_task.id,
            step_id=step_id,
            _external=True,
        )

        # all tasks need to know about db id to load the db entry
        task: chain = load_base.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=1,  # include this step
            prog_target=3,  # select base + select 1 join + select attributes = 3
            prog_unit="",  # unitless counter
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@JOIN_BLP.route("/<int:db_id>/add-join/")
class AddJoinView(MethodView):
    """Add entities that will be joined to the base."""

    @JOIN_BLP.arguments(DataJoinJoinParametersSchema(unknown=EXCLUDE), location="form")
    @JOIN_BLP.response(HTTPStatus.SEE_OTHER)
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Add entities that will be joined to the base."""
        db_task: ProcessingTask | None = ProcessingTask.get_by_id(id_=db_id)

        if not db_task or db_task.task_name not in (
            load_base.name,
            add_data_to_join.name,
        ):
            abort(HTTPStatus.NOT_FOUND)

        # next step
        previous_steps = len(db_task.steps)
        step_id = f"add-join-{previous_steps + 1}"
        href = url_for(
            f"{JOIN_BLP.name}.{FinishJoinsView.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{JOIN_BLP.name}.{AddJoinMicroFrontend.__name__}",
            db_id=db_task.id,
            step_id=step_id,
            _external=True,
        )

        nr_of_joins = len(db_task.data.get("joins", []))

        allwoed_attrs = db_task.data.get("attributes", [])
        if arguments["attribute"] not in allwoed_attrs:
            abort(HTTPStatus.BAD_REQUEST, "Cannot join to a nonexistent attribute!")

        db_task.clear_previous_step()
        db_task.task_name = add_data_to_join.name
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = add_data_to_join.s(  # TODO: call this synchronously? or inline code
            db_id=db_task.id,
            entity_url=arguments["join"],
            join_attr=arguments["attribute"],
        ) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=nr_of_joins + 1,  # include base step in counts
            prog_target=nr_of_joins + 2,  # always require an extra step
            prog_unit="",  # unit less counter
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@JOIN_BLP.route("/<int:db_id>/finish-join/")
class FinishJoinsView(MethodView):
    """Finish selecting joins and proceed to attribute selection."""

    @JOIN_BLP.arguments(
        DataJoinFinishJoinParametersSchema(unknown=EXCLUDE), location="form"
    )
    @JOIN_BLP.response(HTTPStatus.SEE_OTHER)
    @JOIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Finish selecting joins and proceed to attribute selection."""
        db_task: ProcessingTask | None = ProcessingTask.get_by_id(id_=db_id)

        if not db_task or db_task.task_name == add_data_to_join.name:
            abort(HTTPStatus.NOT_FOUND)

        nr_of_joins = len(db_task.data.get("joins", []))

        if nr_of_joins == 0:
            abort(
                HTTPStatus.NOT_FOUND,
                "Cannot finish joins until at least one entity to join is selected.",
            )

        db_task.parameters = DataJoinFinishJoinParametersSchema().dumps(arguments)

        # next step
        next_step_id = "select-attributes"
        href = url_for(
            f"{JOIN_BLP.name}.{AttributeSelectionView.__name__}",
            db_id=db_task.id,
            step_id=next_step_id,
            _external=True,
        )
        ui_href = url_for(
            f"{JOIN_BLP.name}.{AttributeSelectionMicroFrontend.__name__}",
            db_id=db_task.id,
            step_id=next_step_id,
            _external=True,
        )

        db_task.clear_previous_step()
        db_task.save(commit=True)

        app = current_app._get_current_object()
        TASK_STEPS_CHANGED.send(app, task_id=db_id)

        task: Signature = add_step.s(
            task_log="Selecting attributes for final output.",
            db_id=db_task.id,
            step_id=next_step_id,
            href=href,
            ui_href=ui_href,
            prog_value=nr_of_joins + 1,
            prog_target=nr_of_joins + 2,
            prog_unit="",  # unit less counter
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

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
        db_task: ProcessingTask | None = ProcessingTask.get_by_id(id_=db_id)

        if not db_task or db_task.task_name == add_data_to_join.name:
            abort(HTTPStatus.NOT_FOUND)

        nr_of_joins = len(db_task.data.get("joins", []))

        if nr_of_joins == 0:
            abort(
                HTTPStatus.NOT_FOUND,
                "Cannot finish joins until at least one entity to join is selected.",
            )

        db_task.parameters = DataJoinFinishJoinParametersSchema().dumps(arguments)

        db_task.clear_previous_step()
        db_task.task_name = join_data.name
        db_task.save(commit=True)

        app = current_app._get_current_object()
        TASK_STEPS_CHANGED.send(app, task_id=db_id)

        # all tasks need to know about db id to load the db entry
        task: chain = join_data.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
