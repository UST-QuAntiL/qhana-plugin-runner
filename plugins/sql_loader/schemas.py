# Copyright 2023 QHAna plugin runner contributors.
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

from marshmallow import post_load
import marshmallow as ma
from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
)

from .backend.db_enum import DBEnum

from dataclasses import dataclass


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class FirstInputParameters:
    db_enum: DBEnum
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_database: str

    def __str__(self):
        variables = self.__dict__.copy()
        variables["db_password"] = ""
        return str(variables)


@dataclass(repr=False)
class SecondInputParameters:
    custom_query: bool = False
    db_query: str = ""
    table_name: str = ""
    columns_list: str = ""
    save_table: bool = False
    id_attribute: str = ""

    def __str__(self):
        variables = self.__dict__.copy()
        variables["db_password"] = ""
        return str(variables)


class FirstInputParametersSchema(FrontendFormBaseSchema):
    db_enum = EnumField(
        DBEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Database type",
            "description": "Determines the type of database, e.g. MySQL, SQLite, etc.<br>"
            "If ``auto`` is selected, then the plugin tries to resolve this itself. In the case of "
            "``auto``, not every field needs to be filled out, depending on the database. Thus, you "
            "should always try to submit, even if you are uncertain, if the provided information is "
            "sufficient.",
            "input_type": "select",
        },
    )
    db_host = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "DB host",
            "description": "Host of the database.",
            "input_type": "text",
        },
    )
    db_port = ma.fields.Integer(
        required=False,
        allow_none=True,
        metadata={
            "label": "DB port",
            "description": "Port of the database.",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=-1, min_inclusive=True),
    )
    db_user = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "DB user name",
            "description": "The user name for the database.",
            "input_type": "text",
        },
    )
    db_password = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "DB passwort",
            "description": "Password for the database user.",
            "input_type": "password",
        },
    )
    db_database = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "DB database",
            "description": "Name of the database. "
            "In the case of SQLite, this parameter should be the path to the database file.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> FirstInputParameters:
        return FirstInputParameters(**data)


class SecondInputParametersSchema(FrontendFormBaseSchema):
    custom_query = ma.fields.Boolean(
        required=False,
        allow_none=True,
        metadata={
            "label": "Use custom query",
            "description": "If checked, a custom database query can be used.",
            "input_type": "checkbox",
        },
    )
    db_query = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "DB query",
            "description": "The query to be executed on the database.",
            "input_type": "text",
        },
    )
    table_name = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Table",
            "description": "Select the table you want to save.",
            "input_type": "ul",
        },
    )
    columns_list = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Columns",
            "description": "Select the columns you want to keep.",
            "input_type": "ul",
        },
    )
    save_table = ma.fields.Boolean(
        required=False,
        allow_none=True,
        metadata={
            "label": "Save queried table",
            "description": "Saves the queried table as a csv for further use.",
            "input_type": "checkbox",
        },
    )
    id_attribute = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "ID attribute",
            "description": "This determines the attribute that should be used as the ID foreach entity. "
            "If the attribute is not unique for each entry in the queried table, then the index will "
            "be used as the entity's id.",
            "input_type": "search",
            # "list": "id_attribute_list", Not working?
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> SecondInputParameters:
        print(f"loading data: {data}")
        return SecondInputParameters(**data)
