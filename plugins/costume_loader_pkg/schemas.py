from enum import Enum
from typing import List, Any, Optional, Mapping, Type

import marshmallow as ma
from marshmallow import fields, post_load
from marshmallow.utils import resolve_field_instance

from plugins.costume_loader_pkg.backend.aggregator import AggregatorType
from plugins.costume_loader_pkg.backend.attribute import Attribute
from plugins.costume_loader_pkg.backend.attributeComparer import AttributeComparerType
from plugins.costume_loader_pkg.backend.elementComparer import ElementComparerType
from plugins.costume_loader_pkg.backend.entity import Entity
from plugins.costume_loader_pkg.backend.entityComparer import EmptyAttributeAction
from plugins.costume_loader_pkg.backend.entityService import Subset
from plugins.costume_loader_pkg.backend.transformer import TransformerType

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
    def __init__(
        self,
        aggregator: AggregatorType,
        transformer: TransformerType,
        attributes: List[Attribute],
        element_comparers: List[ElementComparerType],
        attribute_comparers: List[AttributeComparerType],
        empty_attribute_actions: List[EmptyAttributeAction],
        filters: List[str] = None,
        amount: int = None,
        subset: Subset = None,
    ):
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
            "options": {"": "–"},
        },
    )
    transformer = EnumField(
        TransformerType,
        required=True,
        metadata={
            "label": "Transformer",
            "description": "Transformer.",
            "input_type": "select",
            "options": {"": "–"},
        },
    )
    attributes = CSVList(
        EnumField(Attribute),
        required=True,
        metadata={
            "label": "Attributes",
            "description": "List of attributes.",
            "input_type": "textarea",
        },
    )
    element_comparers = CSVList(
        EnumField(ElementComparerType),
        required=True,
        metadata={
            "label": "Element comparers",
            "description": "List of element comparers.",
            "input_type": "textarea",
        },
    )
    attribute_comparers = CSVList(
        EnumField(AttributeComparerType),
        required=True,
        metadata={
            "label": "Attribute comparers",
            "description": "A list of attribute comparers.",
            "input_type": "textarea",
        },
    )
    empty_attribute_actions = CSVList(
        EnumField(EmptyAttributeAction),
        required=True,
        metadata={
            "label": "Empty attribute actions",
            "description": "List of empty attribute actions.",
            "input_type": "textarea",
        },
    )
    filters = CSVList(
        fields.Str(),
        allow_none=True,
        metadata={
            "label": "Filters",
            "description": "List of filters.",
            "input_type": "textarea",
        },
    )
    amount = fields.Int(
        allow_none=True,
        metadata={
            "label": "Amount",
            "description": "Amount of costumes to load.",
            "input_type": "textarea",
        },
    )
    subset = EnumField(
        Subset,
        allow_none=True,
        metadata={
            "label": "Subset",
            "description": "Subset to load.",
            "input_type": "select",
            "options": {"": "–"},
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


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class CostumeLoaderUIResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)
