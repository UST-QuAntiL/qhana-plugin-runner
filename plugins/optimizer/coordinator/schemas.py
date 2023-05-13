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


class OptimizerCallbackTaskInputSchema(FrontendFormBaseSchema):
    input_str = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Input String",
            "description": "A simple string input.",
            "input_type": "textarea",
        },
    )


class OptimizerSetupTaskInputSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="*",
        data_content_types=["text/csv"],
        metadata={
            "label": "Dataset URL",
            "description": "URL to a csv file with optimizable data.",
        },
    )
    target_variable = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Target Variable",
            "description": "Name of the target variable in the dataset.",
            "input_type": "text",
        },
    )
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
