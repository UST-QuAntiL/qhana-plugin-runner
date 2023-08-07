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

"""Module to hold DB constant to avoid circular imports."""

from typing import Type, cast
from flask_sqlalchemy.model import DefaultMeta, Model
from sqlalchemy.schema import MetaData
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.orm.decl_api import registry
from blinker import Namespace


DB_SIGNALS = Namespace()


DB: SQLAlchemy = SQLAlchemy(
    metadata=MetaData(
        naming_convention={
            "pk": "pk_%(table_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(column_0_name)s",
        }
    )
)


# Model constant to be importable directly
MODEL = cast(Type[Model], DB.Model)
if type(MODEL) is not DefaultMeta or not issubclass(MODEL, Model):
    raise Warning(
        f"Please update the type cast of db.MODEL to reflect the current type {type(MODEL)}."
    )

# only for sqlalchemy 1.4!
REGISTRY = cast(registry, MODEL.registry)

MIGRATE = Migrate()
