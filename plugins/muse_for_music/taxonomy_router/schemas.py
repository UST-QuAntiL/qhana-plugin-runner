import marshmallow as ma
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, FileUrl

class InputParametersSchema(FrontendFormBaseSchema):
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/list",
        data_content_types=["text/csv", "application/json"],
        metadata={
            "label": "Entities URL",
            "description": "URL to the entity list (e.g., subparts.csv).",
            "input_type": "text",
        },
    )
    taxonomies_zip_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="graph/taxonomy",
        data_content_types=["application/zip"],
        metadata={
            "label": "Taxonomies ZIP URL",
            "description": "URL to the taxonomies.zip file.",
            "input_type": "text",
        },
    )