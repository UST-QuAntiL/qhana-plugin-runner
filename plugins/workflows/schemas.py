import enum
from collections import OrderedDict
from typing import Dict

import marshmallow as ma
from marshmallow import INCLUDE, post_load

from qhana_plugin_runner.api import EnumField, MaBaseSchema
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema

from . import Workflows

config = Workflows.instance.config


class WorkflowsResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)
    pluginType = ma.fields.String(required=False, allow_none=True, dump_only=True)
    tags = ma.fields.List(
        ma.fields.String(), required=False, allow_none=True, dump_only=True
    )


class WorkflowsTaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        input_bpmn: str,
    ):
        self.input_bpmn = input_bpmn


class WorkflowsParametersSchema(FrontendFormBaseSchema):
    input_bpmn = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Input BPMN model name",
            "description": "BPMN model to run.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


class AnyInputSchema(FrontendFormBaseSchema):
    def __init__(self, params=None):
        super().__init__(unknown=INCLUDE)

        if params is None:
            params = {}

        self.inputs: Dict = {}

        self.prefix_choices = config["qhana_input"]["prefix_value_choice"]
        self.prefix_enum = config["qhana_input"]["prefix_value_enum"]
        self.prefix_file_url = config["qhana_input"]["prefix_value_file_url"]
        self.prefix_delimiter = config["qhana_input"]["prefix_value_delimiter"]

        for key, val in params.items():
            if not val["value"]:
                self.add_string_field(key)
            elif val["type"] == "String":
                if val["value"].startswith(
                    f"{self.prefix_choices}{self.prefix_delimiter}"
                ):
                    self.add_choices_field(val, key)
                elif val["value"].startswith(
                    f"{self.prefix_enum}{self.prefix_delimiter}"
                ):
                    self.add_enum_field(val, key)
                elif val["value"].startswith(
                    f"{self.prefix_file_url}{self.prefix_delimiter}"
                ):
                    self.add_file_url_field(val, key)
                else:
                    self.add_string_field(key)

        self.fields = OrderedDict(self.inputs)

    def add_string_field(self, key: str):
        self.inputs[key] = ma.fields.String(
            required=True,
            allow_none=False,
            metadata={
                "label": f"{key}",
                "description": f"Workflow Input {key}",
                "input_type": "text",
            },
        )
        self.__setattr__(key, self.inputs[key])

    def add_choices_field(self, val: Dict, key: str):
        choices = (
            val["value"][
                len(self.prefix_choices) + len(self.prefix_delimiter) : len(val["value"])
            ]
        ).split(",")
        choices_dict = {}
        for choice in choices:
            choices_dict[choice] = choice
        choices_enum = enum.Enum("Enum", choices_dict)
        self.inputs[key] = EnumField(
            choices_enum,
            required=True,
            allow_none=False,
            metadata={
                "label": f"{key}",
                "description": f"Workflow Input {key}",
                "input_type": "select",
            },
        )
        self.__setattr__(key, self.inputs[key])

    def add_enum_field(self, val: Dict, key: str):
        choices = (
            val["value"][
                len(self.prefix_enum) + len(self.prefix_delimiter) : len(val["value"])
            ]
        ).split(",")
        choices_dict = {}
        for choice in choices:
            enum_key, enum_val = choice.split(":")
            choices_dict[enum_key.strip()] = enum_val.strip()
        choices_enum = enum.Enum("Enum", choices_dict)
        self.inputs[key] = EnumField(
            choices_enum,
            required=True,
            allow_none=False,
            metadata={
                "label": f"{key}",
                "description": f"Workflow Input {key}",
                "input_type": "select",
            },
        )
        self.__setattr__(key, self.inputs[key])

    def add_file_url_field(self, val: Dict, key: str):
        data_input_type, data_content_types = (
            val["value"][
                len(self.prefix_file_url) + len(self.prefix_delimiter) : len(val["value"])
            ]
        ).split(",")
        self.inputs[key] = FileUrl(
            required=True,
            allow_none=False,
            data_input_type=data_input_type,
            data_content_types=data_content_types,
            metadata={
                "label": f"{key}",
                "description": f"Workflow Input {key}",
                "input_type": "text",
            },
        )
