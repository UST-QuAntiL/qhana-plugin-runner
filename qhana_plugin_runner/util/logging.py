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

from logging import Logger, getLogger
from flask import Flask


def get_logger(app: Flask, name: str) -> Logger:
    """Utitlity method to get a specific logger that is a child logger of the app.logger."""
    logger_name = f"{app.import_name}.{name}"
    return getLogger(logger_name)


def redact_log_data(
    data: dict, private_fields: tuple = ("ibmq_token", "ibmqToken", "db_password")
):
    """Returns a copy of the data without confidential/private information.

    Args:
        data: The data to be copied.
    Returns:
        A copy of the data without confidential/private information.
    """
    log_data = data.copy()
    for field in private_fields:
        if field in log_data:
            log_data[field] = "****"
    return log_data
