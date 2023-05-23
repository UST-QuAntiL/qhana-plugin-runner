import marshmallow as ma

from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema


class RidgeLossTaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class HyperparamterInputSchema(FrontendFormBaseSchema):
    alpha = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Aplha",
            "description": "Alpha variable for Ridge Loss function.",
            "input_type": "textarea",
        },
    )
