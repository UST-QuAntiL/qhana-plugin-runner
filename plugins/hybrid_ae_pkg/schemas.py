import marshmallow as ma

from qhana_plugin_runner.api import MaBaseSchema
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, FileUrl


class HybridAutoencoderTaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class HybridAutoencoderPennylaneRequestSchema(FrontendFormBaseSchema):
    input_data = FileUrl(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Input Data",
            "description": "URL to the input data.",
            "input_type": "text",
        },
    )
    number_of_qubits = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Number of Qubits",
            "description": "Number of qubits that will be used.",
            "input_type": "text",
        },
    )
    embedding_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Embedding Size",
            "description": "Size the embeddings will have (number of values).",
            "input_type": "text",
        },
    )
    qnn_name = ma.fields.String(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "QNN Name",
            "description": "Name of the QNN that will be used.",
            "input_type": "text",
        },
    )
    training_steps = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Training Steps",
            "description": "Number of training steps",
            "input_type": "text",
        },
    )
