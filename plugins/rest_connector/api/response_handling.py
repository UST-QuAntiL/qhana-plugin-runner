# Copyright 2023 QHAna plugin runner contributors.
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


from logging import Logger
from typing import Dict, Union

from requests import Response

from qhana_plugin_runner.db.models.tasks import ProcessingTask


class ResponseHandlingStrategy:
    __strategies: Dict[str, "ResponseHandlingStrategy"] = {}

    # options that have to be passed to the initial request
    follow_redirects: bool = True
    stream_response: bool = False
    timeout: Union[float, int] = 20

    def __init_subclass__(cls, strategy: str) -> None:
        ResponseHandlingStrategy.__strategies[strategy] = cls()

    @staticmethod
    def get(strategy: str):
        return ResponseHandlingStrategy.__strategies.get(
            strategy,
            ResponseHandlingStrategy.__strategies.get(
                "default", ResponseHandlingStrategy()
            ),
        )

    def handle_response(
        self, response: Response, task: ProcessingTask, logger: Logger
    ) -> Response:
        return response


class DefaultResponseStrategy(ResponseHandlingStrategy, strategy="default"):
    def handle_response(
        self, response: Response, task: ProcessingTask, logger: Logger
    ) -> Response:
        if not response.ok:
            task.add_task_log_entry(
                f"Response status code {response.status_code} indicates an error. Reason: {response.reason}; Error body: {response.text}",
                commit=True,
            )
            response.raise_for_status()
        return response
