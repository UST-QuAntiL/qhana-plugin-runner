from marshmallow import post_load, validate
import marshmallow as ma

from dataclasses import dataclass
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, FileUrl

from .backend.quantum.pl import QNNEnum


@dataclass(repr=False)
class InputParameters:
    train_points_url: str
    test_points_url: str
    number_of_qubits: int
    embedding_size: int
    qnn_name: QNNEnum
    training_steps: int

    def __str__(self):
        variables = self.__dict__.copy()
        return str(variables)


class HybridAutoencoderPennylaneRequestSchema(FrontendFormBaseSchema):
    train_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Training Entity points URL",
            "description": "URL to a json file with the entity points used to train the autoencoder.",
            "input_type": "text",
        },
    )
    test_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Test Entity points URL",
            "description": "URL to a json file with the entity points that should be used for testing. These points will be transformed by the trained autoencoder.",
            "input_type": "text",
        },
    )
    number_of_qubits = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Qubits",
            "description": "The total number of qubits used for the quantum-neural-network.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
    )
    embedding_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Embedding Size",
            "description": "The dimensionality of the embedding (number of values).",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
    )
    qnn_name = EnumField(
        QNNEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "QNN Name",
            "description": "The name of the quantum-neural-network to use as the autoencoder.",
            "input_type": "select",
        },
    )
    training_steps = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Training Steps",
            "description": "The number of training steps to train the autoencoder.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)

    @ma.validates_schema
    def validate_kernel_and_entity_points_urls(self, data, **kwargs):
        # complex errors: Depending on the case, either kernelUrl is not None or entityPointsUrl
        if data:
            number_of_qubits = data.get("number_of_qubits", None)
            embedding_size = data.get("embedding_size", None)
            if number_of_qubits is not None and embedding_size is not None:
                if embedding_size > number_of_qubits:
                    raise ma.ValidationError("The number of qubits must be greater or equal to the embedding size.")
