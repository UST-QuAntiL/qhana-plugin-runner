import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api import MaBaseSchema
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
