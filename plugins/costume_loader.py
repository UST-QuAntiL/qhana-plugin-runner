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
import json
from enum import Enum
from http import HTTPStatus
from typing import Mapping, Optional, Type, Any, List

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE, fields, post_load
from marshmallow.utils import resolve_field_instance
from qhana.backend.aggregator import AggregatorType
from qhana.backend.attribute import Attribute
from qhana.backend.attributeComparer import AttributeComparerType
from qhana.backend.database import Database
from qhana.backend.elementComparer import ElementComparerType
from qhana.backend.entity import Entity
from qhana.backend.entityComparer import EmptyAttributeAction
from qhana.backend.entityService import Subset, EntityService
from qhana.backend.transformer import TransformerType
from sqlalchemy.sql.expression import select

from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "costume-loader"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

COSTUME_LOADER_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Costume loader API.",
    template_folder="costume_loader_templates",
)


class DemoResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class EnumField(fields.Field):
    def __init__(self, enum_type: Type[Enum], **kwargs):
        super().__init__(**kwargs)
        self.enum_type: Type[Enum] = enum_type

    def _serialize(self, value: Enum, attr: str, obj, **kwargs):
        if value is None:
            return None

        return value.value

    def _deserialize(
        self,
        value: str,
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs
    ):
        return self.enum_type(value)


class CSVList(fields.Field):
    def __init__(self, element_type: fields.Field, **kwargs):
        super().__init__(**kwargs)
        self.element_type = resolve_field_instance(element_type)

    def _serialize(self, value: List[Any], attr: str, obj: Any, **kwargs) -> str:
        return ",".join([self.element_type._serialize(v, attr, obj, **kwargs) for v in value])

    def _deserialize(self, value: str, attr: Optional[str], data: Optional[Mapping[str, Any]], **kwargs):
        return [self.element_type.deserialize(v) for v in value.split(",")]


class InputParameters:
    def __init__(
            self, aggregator: AggregatorType, transformer: TransformerType, attributes: List[Attribute],
            element_comparers: List[ElementComparerType], attribute_comparers: List[AttributeComparerType],
            empty_attribute_actions: List[EmptyAttributeAction], filters: List[str] = None, amount: int = None,
            subset: Subset = None):
        self.aggregator = aggregator
        self.transformer = transformer
        self.attributes = attributes
        self.element_comparers = element_comparers
        self.attribute_comparers = attribute_comparers
        self.empty_attribute_actions = empty_attribute_actions
        self.filters = filters
        self.amount = amount
        self.subset = subset


class InputParametersSchema(FrontendFormBaseSchema):
    aggregator = EnumField(
        AggregatorType,
        required=True,
        allow_none=False,
        metadata={
            "label": "Aggregator",
            "description": "Aggregator.",
            "input_type": "select",
            "options": ["mean", "median", "max", "min"]
        })
    transformer = EnumField(
        TransformerType,
        required=True,
        metadata={
            "label": "Transformer",
            "description": "Transformer.",
            "input_type": "select",
            "options": ["linearInverse", "exponentialInverse", "gaussianInverse", "polynomialInverse", "squareInverse"]
        })
    attributes = CSVList(
        EnumField(Attribute),
        required=True,
        metadata={
            "label": "Attributes",
            "description": "List of attributes.",
            "input_type": "textarea",
        })
    element_comparers = CSVList(
        EnumField(ElementComparerType),
        required=True,
        metadata={
            "label": "Element comparers",
            "description": "List of element comparers.",
            "input_type": "textarea",
        })
    attribute_comparers = CSVList(
        EnumField(AttributeComparerType),
        required=True,
        metadata={
            "label": "Attribute comparers",
            "description": "A list of attribute comparers.",
            "input_type": "textarea",
        })
    empty_attribute_actions = CSVList(
        EnumField(EmptyAttributeAction),
        required=True,
        metadata={
            "label": "Empty attribute actions",
            "description": "List of empty attribute actions.",
            "input_type": "textarea",
        })
    filters = CSVList(
        fields.Str(),
        allow_none=True,
        metadata={
            "label": "Filters",
            "description": "List of filters.",
            "input_type": "textarea",
        })
    amount = fields.Int(
        allow_none=True,
        metadata={
            "label": "Amount",
            "description": "Amount of costumes to load.",
            "input_type": "textarea",
        })
    subset = EnumField(
        Subset,
        allow_none=True,
        metadata={
            "label": "Subset",
            "description": "Subset to load.",
            "input_type": "select",
            "options": ["Subset5", "Subset10", "Subset20", "Subset40"]
        })

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


class MuseEntitySchema(MaBaseSchema):
    id = fields.Int(required=True)
    name = fields.Str(required=True)
    kostuem_id = fields.Int(required=True, attribute="kostuemId")
    rollen_id = fields.Int(required=True, attribute="rollenId")
    film_id = fields.Int(required=True, attribute="filmId")
    attributes = fields.List(EnumField(Attribute), required=True)
    values = fields.Dict(EnumField(Attribute), fields.List(fields.Str()), required=True)

    @post_load
    def make_muse_entity(self, data, **kwargs) -> Entity:
        entity = Entity(data["name"])
        entity.set_id(data["id"])
        entity.set_kostuem_id(data["kostuemId"])
        entity.set_rollen_id(data["rollenId"])
        entity.set_film_id(data["filmId"])

        for attr in data["attributes"]:
            entity.add_attribute(attr)

        for k, v in data["values"].items():
            entity.add_value(k, v)

        return entity


class CostumesResponseSchema(MaBaseSchema):
    muse_entities = ma.fields.List(ma.fields.Nested(MuseEntitySchema))


@COSTUME_LOADER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @COSTUME_LOADER_BLP.response(HTTPStatus.OK, DemoResponseSchema())
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

    example_inputs = {
        "aggregator": "mean",
        "transformer": "squareInverse",
        "attributes": "dominanteFarbe",
        "elementComparers": "wuPalmer",
        "attributeComparers": "symMaxMean",
        "emptyAttributeActions": "ignore",
        "filters": "",
        "amount": 0,
        "subset": "Subset5"
    }

    @COSTUME_LOADER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the costume loader plugin."
    )
    @COSTUME_LOADER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @COSTUME_LOADER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the costume loader plugin."
    )
    @COSTUME_LOADER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "costume_loader_template.html",
                name=CostumeLoader.instance.name,
                version=CostumeLoader.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{COSTUME_LOADER_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{COSTUME_LOADER_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@COSTUME_LOADER_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @COSTUME_LOADER_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @COSTUME_LOADER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: InputParametersSchema):
        """Start the costume loading task."""
        db_task = ProcessingTask(task_name=costume_loading_task.name, parameters=InputParametersSchema().dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = costume_loading_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return {
            "name": costume_loading_task.name,
            "task_id": str(result.id),
            "task_result_url": url_for("tasks-api.TaskView", task_id=str(result.id)),
        }


class CostumeLoader(QHAnaPluginBase):
    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return COSTUME_LOADER_BLP

    def get_requirements(self) -> str:
        return "git+ssh://git@github.com/UST-QuAntiL/qhana.git@95fb144eafd8ba105594358d93ed09a36254ff41#egg=qhana"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{CostumeLoader.instance.identifier}.costume_loading_task", bind=True)
def costume_loading_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    param_schema = InputParametersSchema()
    input_params: InputParameters = param_schema.loads(task_data.parameters)

    es = EntityService()

    plan = [
        input_params.aggregator,
        input_params.transformer,
    ]
    plan.extend([(
        input_params.attributes[i],
        input_params.element_comparers[i],
        input_params.attribute_comparers[i],
        input_params.empty_attribute_actions[i],
        input_params.filters[i]) for i in range(len(input_params.attributes))])

    es.add_plan(plan)

    db = Database()
    db.open("plugins/config.ini")

    es.create_subset(input_params.subset, db)

    entity_schema = MuseEntitySchema()

    return entity_schema.dumps(es.allEntities[0])
