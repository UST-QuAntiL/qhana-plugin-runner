# Copyright 2026 QHAna plugin runner contributors.
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

from logging import INFO

import pytest

from qhana_plugin_runner import create_app
from qhana_plugin_runner.db.cli import create_db_function
from qhana_plugin_runner.db.models.tasks import ProcessingTask


TEST_CONFIG = {
    "SECRET_KEY": "test",
    "DEBUG": False,
    "TESTING": True,
    "JSON_SORT_KEYS": True,
    "JSONIFY_PRETTYPRINT_REGULAR": False,
    "DEFAULT_LOG_SEVERITY": INFO,
    "DEFAULT_LOG_FORMAT_STYLE": "{",
    "DEFAULT_LOG_FORMAT": "{asctime} [{levelname:^7}] [{module:<30}] {message}    <{funcName}, {lineno}; {pathname}>",
    "OPENAPI_VERSION": "3.0.2",
    "OPENAPI_JSON_PATH": "api-spec.json",
    "OPENAPI_URL_PREFIX": "",
    "DEFAULT_FILE_STORE": "local_filesystem",
    "FILE_STORE_ROOT_PATH": "files",
    "PLUGIN_FOLDERS": ["./plugins/qiskit_executor"],
    "DISABLED_PLUGINS": [],
}


@pytest.fixture()
def qiskit_executor_client(tmp_path):
    test_config = dict(TEST_CONFIG)
    test_config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{tmp_path / 'qiskit-executor.db'}"

    app = create_app(test_config, silent_log=True)

    with app.app_context():
        create_db_function(app)
        db_task = ProcessingTask(
            task_name="qiskit-executor-test",
            data={
                "parameters": {"backend": "ibmq_qasm_simulator"},
                "backend_names": ["ibmq_qasm_simulator", "ibm_brisbane"],
            },
        )
        db_task.save(commit=True)

    return app.test_client()


def test_authentication_step_ui_get_renders_with_incomplete_task_state(
    qiskit_executor_client,
):
    response = qiskit_executor_client.get(
        "/plugins/qiskit-executor@v0-1-1/1/authentication-step-ui/"
    )

    assert response.status_code == 200
    assert "ibmq_qasm_simulator" in response.get_data(as_text=True)


def test_authentication_step_ui_post_renders_without_argument_binding_error(
    qiskit_executor_client,
):
    response = qiskit_executor_client.post(
        "/plugins/qiskit-executor@v0-1-1/1/authentication-step-ui/",
        data={"ibmqToken": "test-token", "backend": "ibmq_qasm_simulator"},
    )

    assert response.status_code == 200
    assert response.mimetype == "text/html"


def test_backend_selection_step_ui_post_renders_without_argument_binding_error(
    qiskit_executor_client,
):
    response = qiskit_executor_client.post(
        "/plugins/qiskit-executor@v0-1-1/1/backend-selection-ui/",
        data={"backend": "ibmq_qasm_simulator"},
    )

    assert response.status_code == 200
    assert "ibmq_qasm_simulator" in response.get_data(as_text=True)
