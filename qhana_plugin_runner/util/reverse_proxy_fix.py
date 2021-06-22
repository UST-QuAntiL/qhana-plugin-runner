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

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix


def apply_reverse_proxy_fix(app: Flask):
    """Apply the reverse proxy fix from werkzeug with the number configured in REVERSE_PROXY_COUNT."""
    r_p_count = app.config.get("REVERSE_PROXY_COUNT", 0)
    if r_p_count > 0:
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=r_p_count,
            x_host=r_p_count,
            x_port=r_p_count,
            x_prefix=r_p_count,
            x_proto=r_p_count,
        )
