import marshmallow as ma

from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema


class CallbackURLSchema(MaBaseSchema):
    callback_url = ma.fields.URL(required=True, allow_none=False)
