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

"""Tests for the db module of the plugin_utils."""

from os import environ
import pytest
from dotenv import load_dotenv

load_dotenv(".flaskenv")
load_dotenv(".env")

MODULE_NAME = "qhana_plugin_runner"

from qhana_plugin_runner import create_app
from qhana_plugin_runner.db.cli import create_db_function, drop_db, create_db

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.mutable_json import (
    NestedMutableDict,
    NestedMutableList,
)


@pytest.fixture(scope="function")
def task_data():

    environ["SQLALCHEMY_DATABASE_URI"] = "sqlite:///../instance/test.db"
    app = create_app()
    with app.app_context():
        try:
            drop_db()
        except:
            pass
        create_db_function(app)
        task_data = ProcessingTask(task_name="test-data")
        task_data.save(commit=True)
        yield task_data
