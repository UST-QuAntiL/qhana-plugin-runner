import marshmallow as ma

from qhana_plugin_runner.api import MaBaseSchema
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, FileUrl


class HybridAutoencoderPennylaneRequestSchema(FrontendFormBaseSchema):
    input_data = FileUrl(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Input Data",
            "description": "A URL to the data input file. Example: `data:text/plain,0,0,0,0,0,0,0,0,0,0` ",
            "input_type": "text",
        },
    )
    number_of_qubits = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Number of Qubits",
            "description": "The total number of qubits used for the quantum-neural-network. Example: `3`",
            "input_type": "text",
        },
    )
    embedding_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Embedding Size",
            "description": "The dimensionality of the embedding (number of values). Example: `2`",
            "input_type": "text",
        },
    )
    qnn_name = ma.fields.String(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "QNN Name",
            "description": "The name of the quantum-neural-network to use as the autoencoder. Example: `QNN3`",
            "input_type": "text",
        },
    )
    training_steps = ma.fields.Integer(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={
            "label": "Training Steps",
            "description": "The number of training steps to train the autoencoder. Example: `100`",
            "input_type": "text",
        },
    )
