from enum import Enum
from typing import List, Any, Optional, Mapping

import marshmallow as ma
from marshmallow import fields, post_load
from marshmallow.utils import resolve_field_instance

from qhana_plugin_runner.api import MaBaseSchema
from qhana_plugin_runner.api.util import FrontendFormBaseSchema
from qhana_plugin_runner.api.extra_fields import EnumField, CSVList


class CostumeType(Enum):
    WITHOUT_BASE_ELEMENTS = "keine Basiselemente"
    WITH_BASE_ELEMENTS = "Basiselemente"


class InputParameters:
    def __init__(
        self,
        costume_type: CostumeType,
        db_host: str,
        db_user: str,
        db_password: str,
        db_database: str,
    ):
        self.costume_type = costume_type
        self.db_host = db_host
        self.db_user = db_user
        self.db_password = db_password
        self.db_database = db_database


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

    db_host = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "DB host",
            "description": "Host of the mysql database.",
            "input_type": "text",
        },
    )

    db_user = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "DB user name",
            "description": "A user name for the mysql database.",
            "input_type": "text",
        },
    )

    db_password = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "DB password",
            "description": "Password for the database user.",
            "input_type": "password",
        },
    )

    db_database = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "DB database",
            "description": "Name of the mysql database.",
            "input_type": "text",
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
