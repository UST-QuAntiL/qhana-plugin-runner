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

"""Module containing database cli and models."""

from flask import Flask
from sqlalchemy import event
from sqlalchemy.engine import Engine

from .cli import register_db_cli_blueprint
from .db import DB, MIGRATE


def register_db(app: Flask):
    """Register the sqlalchemy db and alembic migrations with the flask app."""
    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        app.config[
            "SQLALCHEMY_DATABASE_URI"
        ] = f"sqlite:///{app.instance_path}/{app.import_name}.db"

    DB.init_app(app)
    app.logger.info(f'Connected to db "{app.config["SQLALCHEMY_DATABASE_URI"]}".')

    register_db_cli_blueprint(app)

    MIGRATE.init_app(app, DB)

    # Apply additional config for Sqlite databases
    if app.config.get("SQLALCHEMY_DATABASE_URI", "").startswith("sqlite://"):
        from sqlite3 import Connection as SQLiteConnectionClass

        from sqlalchemy.dialects.sqlite.base import SQLiteDialect

        @event.listens_for(Engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            is_sqlite: bool = False

            try:
                # Try to get dialect from the private pool (use manual name Wrangling)
                pool = connection_record._ConnectionRecord__pool
                # Test if dialect is SQLite
                is_sqlite = isinstance(pool._dialect, SQLiteDialect)
            except AttributeError:
                pass

            if isinstance(dbapi_connection, SQLiteConnectionClass):
                is_sqlite = True

            if is_sqlite and app.config.get("SQLITE_FOREIGN_KEYS", True):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
