from typing import List, Any, Optional, Mapping

import marshmallow as ma
from marshmallow import fields
from marshmallow.utils import resolve_field_instance

from qhana_plugin_runner.api import MaBaseSchema


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


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class CostumeLoaderUIResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)
