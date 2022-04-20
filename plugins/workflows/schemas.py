import enum

import marshmallow as ma
from collections import OrderedDict

from celery.utils.log import get_task_logger
from flask import current_app
from . import conf as config
from marshmallow import post_load, INCLUDE
from qhana_plugin_runner.api import MaBaseSchema, EnumField
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, FileUrl


class WorkflowsResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)
    pluginType = ma.fields.String(required=False, allow_none=True, dump_only=True)
    tags = ma.fields.List(ma.fields.String(), required=False, allow_none=True, dump_only=True)


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
        }
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


# TODO: Cleanup
class AnyInputSchema(FrontendFormBaseSchema):
    def __init__(self, params=None):
        super().__init__(unknown=INCLUDE)

        if params is None:
            params = {}

        inputs = {}

        for key, val in params.items():
            prefix_choices = config["qhana_input"]["prefix_value_choice"]
            prefix_enum = config["qhana_input"]["prefix_value_enum"]
            prefix_file_url = config["qhana_input"]["prefix_value_file_url"]
            prefix_delimiter = config["qhana_input"]["prefix_value_delimiter"]

            if not val["value"]:
                inputs[key] = ma.fields.String(
                    required=True,
                    allow_none=False,
                    metadata={
                        "label": f"{key}",
                        "description": f"Workflow Input {key}",
                        "input_type": "text",
                    }
                )
                self.__setattr__(key, inputs[key])
                continue
            if val["type"] == "String" and val["value"].startswith(f"{prefix_choices}{prefix_delimiter}"):
                choices = (val["value"][len(prefix_choices) + len(prefix_delimiter):len(val["value"])]).split(",")
                choices_dict = {}
                for choice in choices:
                    choices_dict[choice] = choice
                choices_enum = enum.Enum('Enum', choices_dict)
                inputs[key] = EnumField(
                    choices_enum,
                    required=True,
                    allow_none=False,
                    metadata={
                        "label": f"{key}",
                        "description": f"Workflow Input {key}",
                        "input_type": "select",
                    },
                )
                self.__setattr__(key, inputs[key])
            elif val["type"] == "String" and val["value"].startswith(f"{prefix_enum}{prefix_delimiter}"):
                choices = (val["value"][len(prefix_enum) + len(prefix_delimiter):len(val["value"])]).split(",")
                choices_dict = {}
                for choice in choices:
                    enum_key, enum_val = choice.split(":")
                    choices_dict[enum_key.strip()] = enum_val.strip()
                choices_enum = enum.Enum('Enum', choices_dict)
                inputs[key] = EnumField(
                    choices_enum,
                    required=True,
                    allow_none=False,
                    metadata={
                        "label": f"{key}",
                        "description": f"Workflow Input {key}",
                        "input_type": "select",
                    },
                )
                self.__setattr__(key, inputs[key])
            elif val["type"] == "String" and val["value"].startswith(f"{prefix_file_url}{prefix_delimiter}"):
                data_input_type, data_content_types = \
                    (val["value"][len(prefix_file_url) + len(prefix_delimiter):len(val["value"])]).split(",")
                inputs[key] = FileUrl(
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
            elif val["type"] == "String":
                inputs[key] = ma.fields.String(
                    required=True,
                    allow_none=False,
                    metadata={
                        "label": f"{key}",
                        "description": f"Workflow Input {key}",
                        "input_type": "text",
                    }
                )
                self.__setattr__(key, inputs[key])

        self.fields = OrderedDict(inputs)
