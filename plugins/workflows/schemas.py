import enum

import marshmallow as ma
from collections import OrderedDict

from celery.utils.log import get_task_logger
from flask import current_app
from . import conf as config
from marshmallow import post_load, INCLUDE
from qhana_plugin_runner.api import MaBaseSchema, EnumField
from qhana_plugin_runner.api.util import FrontendFormBaseSchema


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


class AnyInputSchema(FrontendFormBaseSchema):
    def __init__(self, params=None):
        super().__init__(unknown=INCLUDE)
        if params is None:
            params = {}
        inputs = {}
        for key, val in params.items():
            prefix = config["qhana_input"]["prefix_value_choice"]
            prefix_delimiter = config["qhana_input"]["prefix_value_delimiter"]
            if val["type"] == "String" and val["value"].startswith(f"{prefix}{prefix_delimiter}"):
                choices = (val["value"][len(prefix)+len(prefix_delimiter):len(val["value"])]).split(",")
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
