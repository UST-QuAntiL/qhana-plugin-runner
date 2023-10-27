# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mimetypes
from qhana_plugin_runner.plugin_utils.entity_marshalling import ensure_dict, load_entities
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.util.plugins import QHAnaPluginBase


def parse_execution_options(url: str) -> dict:
    """Load execution options from url."""
    with open_url(url) as execution_options_response:
        try:
            mimetype = execution_options_response.headers["Content-Type"]
        except KeyError:
            mimetype = mimetypes.MimeTypes().guess_type(url=url)[0]
        if mimetype is None:
            msg = "Could not guess execution options mime type!"
            QHAnaPluginBase.__app__.logger.error(msg)
            raise ValueError(msg)  # TODO better error
        entities = ensure_dict(
            load_entities(execution_options_response, mimetype=mimetype)
        )
    options = next(entities, {})
    return options
