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


class SmorestProductionConfig:
    OPENAPI_VERSION = "3.0.2"
    OPENAPI_JSON_PATH = "api-spec.json"
    OPENAPI_URL_PREFIX = ""

    # OpenAPI Documentation renderers:
    OPENAPI_REDOC_PATH = "/redoc/"
    OPENAPI_REDOC_URL = (
        "https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"
    )

    OPENAPI_RAPIDOC_PATH = "/rapidoc/"
    OPENAPI_RAPIDOC_URL = "https://cdn.jsdelivr.net/npm/rapidoc/dist/rapidoc-min.js"
    # mor config options: https://mrin9.github.io/RapiDoc/api.html
    OPENAPI_RAPIDOC_CONFIG = {"use-path-in-nav-bar": "true"}

    OPENAPI_SWAGGER_UI_PATH = "/swagger-ui/"
    OPENAPI_SWAGGER_UI_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"


class SmorestDebugConfig(SmorestProductionConfig):
    # do not propagate exceptions in debug mode
    # this makes it hard to test the api and an api client at the same time
    PROPAGATE_EXCEPTIONS = False
