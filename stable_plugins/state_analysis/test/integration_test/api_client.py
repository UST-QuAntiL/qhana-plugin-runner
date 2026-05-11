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

import json
import os
import time

import pytest
import requests

# Default plugin-runner URL for integration tests; override with the
# QHANA_PLUGIN_RUNNER_URL environment variable when running against a
# non-default port or remote host.
DEFAULT_BASE_URL = os.environ.get("QHANA_PLUGIN_RUNNER_URL", "http://localhost:5005")


class APIClient:
    def __init__(
        self,
        plugin_name: str,
        version: str,
        base_url: str = DEFAULT_BASE_URL,
    ):
        """
        Initializes the API client with the base URL, plugin name, and version.
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/plugins/{plugin_name}@{version}/process/"

    def send_request(
        self, data: dict, max_retries: int = 10, wait_seconds: float = 1.0
    ) -> dict:
        """
        Sends a request to the API and follows the redirect to get the response.
        Retries fetching the result if status is 'PENDING'.

        :param data: Dictionary containing the request payload.
        :param max_retries: Maximum number of retries if status remains 'PENDING'.
        :param wait_seconds: Waiting time in seconds between retries.
        :return: Parsed JSON response.
        :raises ValueError: If any request or JSON parsing error occurs or if the
                            status remains 'PENDING' after all retries.
        """
        try:
            # POST + follow redirect to the result URL in one step
            response = requests.post(self.api_url, data=data)
            response.raise_for_status()
            result_url = response.url

            # Poll the result URL until it leaves the PENDING state
            for _ in range(max_retries):
                redirect_res = requests.get(result_url)
                redirect_res.raise_for_status()

                result_data = redirect_res.json()
                if result_data.get("status") != "PENDING":
                    return result_data

                time.sleep(wait_seconds)

            raise ValueError(f"Result is still 'PENDING' after {max_retries} retries.")

        except requests.RequestException as req_err:
            raise ValueError(f"Request error: {req_err}") from req_err
        except (ValueError, json.JSONDecodeError) as json_err:
            raise ValueError(f"JSON error: {json_err}") from json_err
        except Exception as exc:
            raise ValueError(f"Unexpected error: {exc}") from exc
