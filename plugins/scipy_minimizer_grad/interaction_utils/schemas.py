from dataclasses import dataclass

import marshmallow as ma

from qhana_plugin_runner.api.util import MaBaseSchema


@dataclass
class CallbackUrl:
    callback: str


class CallbackUrlSchema(MaBaseSchema):
    callback = ma.fields.URL(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CallbackUrl(**data)
