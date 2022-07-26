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

"""Root module containing the flask app factory."""
import os
import re
from json import load as load_json
from json import loads
from logging import WARNING, Formatter, Handler, Logger, getLogger
from logging.config import dictConfig
from os import environ, makedirs
from typing import IO, Any, Dict, Mapping, Optional, cast

import click
from flask.app import Flask
from flask.cli import FlaskGroup
from flask.config import Config
from flask.logging import default_handler
from flask_cors import CORS
from tomlkit.api import parse as parse_toml

from . import api, babel, celery, db, requests
from .api import jwt_helper
from .licenses import register_licenses
from .markdown import register_markdown_filter
from .plugins_cli import register_plugin_cli_blueprint
from .storage import register_file_store
from .util.config import DebugConfig, ProductionConfig
from .util.jinja_helpers import register_helpers
from .util.plugins import register_plugins
from .util.templates import register_templates
from .util.request_helpers import register_additional_schemas

# change this to change tha flask app name and the config env var prefix
# must not contain any spaces!
APP_NAME = __name__
ENV_VAR_PREFIX = APP_NAME.upper().replace("-", "_").replace(" ", "_")


def load_toml(file_like: IO[Any]) -> Mapping[str, Any]:
    return parse_toml("\n".join(file_like.readlines()))


def create_app(test_config: Optional[Dict[str, Any]] = None):
    """Flask app factory."""

    instance_folder_env_var = f"{ENV_VAR_PREFIX}_INSTANCE_FOLDER"

    # create and configure the app
    app = Flask(
        APP_NAME,
        instance_relative_config=True,
        instance_path=environ.get(instance_folder_env_var, None),
    )

    # Start Loading config #################

    # load defaults
    config = cast(Config, app.config)
    flask_env = cast(Optional[str], config.get("ENV"))
    if flask_env == "production":
        config.from_object(ProductionConfig)
    elif flask_env == "development":
        config.from_object(DebugConfig)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        config.from_pyfile("config.py", silent=True)
        # also try to load json config
        config.from_file("config.json", load=load_json, silent=True)
        # also try to load toml config
        config.from_file("config.toml", load=load_toml, silent=True)
        # load config from file specified in env var
        config.from_envvar(f"{ENV_VAR_PREFIX}_SETTINGS", silent=True)
        # TODO load some config keys directly from env vars

        # load Redis URLs from env vars
        if "BROKER_URL" in os.environ and "RESULT_BACKEND" in os.environ:
            config["CELERY"] = {
                "broker_url": os.environ["BROKER_URL"],
                "result_backend": os.environ["RESULT_BACKEND"],
            }

        if "PLUGIN_FOLDERS" in os.environ:
            config["PLUGIN_FOLDERS"] = [
                folder for folder in os.environ["PLUGIN_FOLDERS"].split(":") if folder
            ]

        if "TEMPLATE_FOLDERS" in os.environ:
            config["TEMPLATE_FOLDERS"] = [
                folder for folder in os.environ["TEMPLATE_FOLDERS"].split(":") if folder
            ]

        # load database URI from env vars
        if "SQLALCHEMY_DATABASE_URI" in os.environ:
            config["SQLALCHEMY_DATABASE_URI"] = os.environ["SQLALCHEMY_DATABASE_URI"]

        if "URL_MAP" in os.environ:
            config["URL_REWRITE_RULES"] = loads(os.environ["URL_MAP"])

        if "DEFAULT_FILE_STORE" in os.environ:
            config["DEFAULT_FILE_STORE"] = os.environ["DEFAULT_FILE_STORE"]
    else:
        # load the test config if passed in
        config.from_mapping(test_config)

    if isinstance(config.get("URL_REWRITE_RULES"), Mapping):
        # rewrite mapping to tuple sequence and precompile regex patterns
        url_map = config["URL_REWRITE_RULES"]
        config["URL_REWRITE_RULES"] = [
            (re.compile(key), value)  # pattern, replacement pairs
            for key, value in url_map.items()
            if isinstance(key, str) and isinstance(value, str)  # only str, str allowed
        ]
        if len(config["URL_REWRITE_RULES"]) != len(url_map):
            pass  # TODO some rewrite rules were dismissed as invalid!

    # End Loading config #################

    # Configure logging
    log_config = cast(Optional[Dict[Any, Any]], config.get("LOG_CONFIG"))
    if log_config:
        # Apply full log config from dict
        dictConfig(log_config)
    else:
        # Apply smal log config to default handler
        log_severity = max(0, config.get("DEFAULT_LOG_SEVERITY", WARNING))
        # use percent for backwards compatibility in case of errors
        log_format_style = cast(str, config.get("DEFAULT_LOG_FORMAT_STYLE", "%"))
        log_format = cast(Optional[str], config.get("DEFAULT_LOG_FORMAT"))
        date_format = cast(Optional[str], config.get("DEFAULT_LOG_DATE_FORMAT"))
        if log_format:
            formatter = Formatter(log_format, style=log_format_style, datefmt=date_format)
            default_logging_handler = cast(Handler, default_handler)
            default_logging_handler.setFormatter(formatter)
            default_logging_handler.setLevel(log_severity)
            root = getLogger()
            root.addHandler(default_logging_handler)
            app.logger.removeHandler(default_logging_handler)

    logger: Logger = app.logger
    logger.info(
        f"Configuration loaded. Instance folder is at path '{app.instance_path}' (can be changed by setting the {instance_folder_env_var} environmen variable)."
    )
    logger.info(
        f"Possible config locations are: 'config.py', 'config.json', Environment: '{ENV_VAR_PREFIX}_SETTINGS'"
    )

    if config.get("SECRET_KEY") == "debug_secret":
        logger.error(
            'The configured SECRET_KEY="debug_secret" is unsafe and must not be used in production!'
        )

    # ensure the instance folder exists
    try:
        makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass

    # Begin loading extensions and routes

    babel.register_babel(app)

    celery.register_celery(app)

    db.register_db(app)

    jwt_helper.register_jwt(app)
    api.register_root_api(app)

    # register jinja helpers
    register_helpers(app)
    register_markdown_filter(app)

    # register request helpers with request session
    register_additional_schemas(requests.REQUEST_SESSION)

    # register plugins, AFTER registering the API!
    register_plugins(app)
    register_plugin_cli_blueprint(app)

    # register templates
    register_templates(app)

    # register file store after plugins to allow plugins to contribute file store implementations
    register_file_store(app)

    # register the licenses blueprint
    register_licenses(app)

    # allow cors requests everywhere (CONFIGURE THIS TO YOUR PROJECTS NEEDS!)
    CORS(app)

    if config.get("DEBUG", False):
        # Register debug routes when in debug mode
        from .util.debug_routes import register_debug_routes

        register_debug_routes(app)

        # register jinja debug extension
        app.jinja_env.add_extension("jinja2.ext.debug")

    return app


@click.group(cls=FlaskGroup, create_app=create_app)
def cli():
    """Cli entry point for autodoc tooling."""
    pass
