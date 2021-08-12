from enum import Enum
from typing import List, Any, Optional, Mapping

import marshmallow as ma
from marshmallow import fields, post_load
from marshmallow.utils import resolve_field_instance

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


class CostumeType(Enum):
    WITHOUT_BASE_ELEMENTS = "keine Basiselemente"
    WITH_BASE_ELEMENTS = "Basiselemente"


class InputParameters:
    def __init__(self, costume_type: CostumeType):
        self.costume_type = costume_type


class InputParametersSchema(FrontendFormBaseSchema):
    costume_type = EnumField(
        CostumeType,
        required=True,
        metadata={
            "label": "Costume Type",
            "description": "Load costumes as one costume per entity or one base element per entity.",
            "input_type": "select",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class CostumeLoaderUIResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)
