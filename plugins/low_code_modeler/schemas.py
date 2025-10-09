
import marshmallow as ma

from qhana_plugin_runner.api.util import MaBaseSchema


class MetadataOfModellSchema(MaBaseSchema):
    """Metadaten von Modellen"""
    id = ma.fields.String()
    version = ma.fields.String()
    name = ma.fields.String()
    date = ma.fields.AwareDateTime()
    autosave = ma.fields.Bool(
        required=False, missing=False, description="If this save was an autosave.")
    model_id = ma.fields.String()


class SaveModelParamsSchema(MaBaseSchema):
    """Parameter vom Speichervorgang"""
    autosave = ma.fields.Bool(required=False, missing=False, description="Set this to true for autosaves.")
