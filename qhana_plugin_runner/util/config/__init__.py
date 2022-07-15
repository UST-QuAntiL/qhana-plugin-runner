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

"""Module containing default config values."""
import re
from logging import INFO, WARNING
from os import urandom
from typing import Sequence, Tuple

from .celery_config import CELERY_DEBUG_CONFIG, CELERY_PRODUCTION_CONFIG
from .smorest_config import SmorestDebugConfig, SmorestProductionConfig
from .sqlalchemy_config import SQLAchemyDebugConfig, SQLAchemyProductionConfig


class ProductionConfig(SQLAchemyProductionConfig, SmorestProductionConfig):
    SECRET_KEY = urandom(32)

    REVERSE_PROXY_COUNT = 0

    DEBUG = False
    TESTING = False

    JSON_SORT_KEYS = True
    JSONIFY_PRETTYPRINT_REGULAR = False

    LOG_CONFIG = None  # if set this is preferred

    DEFAULT_LOG_SEVERITY = WARNING
    DEFAULT_LOG_FORMAT_STYLE = "{"
    DEFAULT_LOG_FORMAT = "{asctime} [{levelname:^7}] [{module:<30}] {message}    <{funcName}, {lineno}; {pathname}>"
    DEFAULT_LOG_DATE_FORMAT = None

    CELERY = CELERY_PRODUCTION_CONFIG

    DEFAULT_FILE_STORE = "local_filesystem"
    FILE_STORE_ROOT_PATH = "files"

    # URL rewrite rules are (pattern, replacement) pairs that are applied
    # in order to URLs opened with qhana_plugin_runner.requests.open_url
    URL_REWRITE_RULES: Sequence[Tuple[re.Pattern, str]] = []


class DebugConfig(ProductionConfig, SQLAchemyDebugConfig, SmorestDebugConfig):
    DEBUG = True
    SECRET_KEY = "debug_secret"  # FIXME make sure this NEVER! gets used in production!!!

    DEFAULT_LOG_SEVERITY = INFO

    CELERY = CELERY_DEBUG_CONFIG

    # TODO allow specifying this as a Environment variable
    PLUGIN_FOLDERS = ["./plugins", "./local_plugins"]
