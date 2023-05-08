import marshmallow as ma
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    PluginUrl,
    FileUrl,
)


class OptimizerTaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class SelectObjecriveFunctionInputSchema(FrontendFormBaseSchema):
    objective_function_plugin_selector = PluginUrl(
        required=True,
        allow_none=False,
        plugin_tags=["objective-function"],
        metadata={
            "label": "Objective-Function Plugin Selector",
            "description": "URL of objective-function-plugin.",
            "input_type": "text",
        },
    )


class SelectDatasetInputSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True,
        allow_none=False,
        metadata={
            "label": "Dataset URL",
            "description": "URL to a csv file with optimizable data.",
            "input_type": "text",
        },
    )
