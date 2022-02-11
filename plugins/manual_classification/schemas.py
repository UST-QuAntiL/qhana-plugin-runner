import marshmallow as ma
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema, FileUrl


class ResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class LoadParametersSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={"label": "Entities URL"},
    )


class ClassificationSchema(FrontendFormBaseSchema):
    class_identifier = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Class Name",
            "description": "Name of the class to be annotated",
            "input_type": "textfield",
        },
    )
