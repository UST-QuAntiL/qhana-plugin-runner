from enum import Enum
from typing import List, Any, Optional, Mapping, Type

import marshmallow as ma
from marshmallow import fields, post_load
from marshmallow.utils import resolve_field_instance

from plugins.costume_loader_pkg.backend.attribute import Attribute
from plugins.costume_loader_pkg.backend.entity import Entity

from qhana_plugin_runner.api import MaBaseSchema
from qhana_plugin_runner.api.util import FrontendFormBaseSchema
from qhana_plugin_runner.api.extra_fields import EnumField


class CSVList(fields.Field):
    def __init__(self, element_type: fields.Field, **kwargs):
        super().__init__(**kwargs)
        self.element_type = resolve_field_instance(element_type)

    def _serialize(self, value: List[Any], attr: str, obj: Any, **kwargs) -> str:
        return ",".join(
            [self.element_type._serialize(v, attr, obj, **kwargs) for v in value]
        )

    def _deserialize(
        self, value: str, attr: Optional[str], data: Optional[Mapping[str, Any]], **kwargs
    ):
        return [self.element_type.deserialize(v) for v in value.split(",")]


class InputParameters:
    def __init__(self, attributes: List[Attribute]):
        self.attributes = attributes


class InputParametersSchema(FrontendFormBaseSchema):
    attributes = CSVList(
        EnumField(Attribute),
        required=True,
        metadata={
            "label": "Attributes",
            "description": "List of attributes.",
            "input_type": "textarea",
        },
    )

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
        entity.id = data["id"]
        entity.kostuemId = data["kostuemId"]
        entity.rollenId = data["rollenId"]
        entity.filmId = data["filmId"]

        for attr in data["attributes"]:
            entity.add_attribute(attr)

        for k, v in data["values"].items():
            entity.add_value(k, v)

        return entity


class CostumesResponseSchema(MaBaseSchema):
    muse_entities = ma.fields.List(ma.fields.Nested(MuseEntitySchema))


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class CostumeLoaderUIResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)
