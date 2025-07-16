import marshmallow as ma

from qhana_plugin_runner.api.util import FrontendFormBaseSchema


class Schema(FrontendFormBaseSchema):
    """Validates two text inputs (QASM code and metadata)."""

    qasmCode = ma.fields.String(
        required=True,
        metadata={
            "label": "qasmCode",
            "description": "QASM code that will be stored as a file.",
            "input_type": "textarea",
        },
    )
