import marshmallow as ma
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema, FileUrl


class ResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class LoadParametersSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True,
        allow_none=False,
        data_content_types=["text/csv", "application/json", "application/X-lines+json"],
        data_input_type="entity",
        metadata={
            "label": "Entities URL",
            "description": "URL to a json file with a list of entities.",
            "input_type": "text",
        },
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
