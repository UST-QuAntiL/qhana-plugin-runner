from dataclasses import dataclass

from qhana_plugin_runner.api.util import MaBaseSchema
import marshmallow as ma


@dataclass
class CallbackURLData:
    callback_url: str


class CallbackURLSchema(MaBaseSchema):
    callback_url = ma.fields.URL(required=True, allow_none=False, data_key="callbackUrl")

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CallbackURLData(**data)