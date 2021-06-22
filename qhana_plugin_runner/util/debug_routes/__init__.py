# Copyright 2021 QHAna plugin runner contributors.
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

# originally from <https://github.com/buehlefs/flask-template/>

"""
Module containing Debug Methods and sites.

This Module should only be loaded in debug Mode.
"""

from flask.app import Flask
from . import root  # noqa
from . import routes  # noqa


def register_debug_routes(app: Flask):
    """Register the debug routes blueprint with the flask app."""
    if not app.config["DEBUG"]:
        app.logger.warning("This Module should only be loaded if DEBUG mode is active!")
        raise Warning("This Module should only be loaded if DEBUG mode is active!")
    app.register_blueprint(root.DEBUG_BLP)
