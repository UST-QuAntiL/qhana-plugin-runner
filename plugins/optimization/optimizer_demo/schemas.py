import marshmallow as ma
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParametersSchema(FrontendFormBaseSchema):
    pass


class DatasetInputSchema(FrontendFormBaseSchema):
    dataset_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="dataset",
        data_content_types="application/json",
        metadata={
            "label": "Dataset URL",
            "description": "URL to a dataset.",
            "input_type": "text",
        },
    )
