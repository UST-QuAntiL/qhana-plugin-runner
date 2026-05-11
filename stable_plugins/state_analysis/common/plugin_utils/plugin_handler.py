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

import requests
from flask import current_app, has_app_context, url_for

# TODO (review): this whole module — and the filesMaker plugin it talks to —
# should be removed.


def _resolve_base_url() -> str:
    """Resolve the plugin runner's base URL from the current request,
    application config, or environment variable.

    Falls back to ``http://localhost:5005`` only as
    a last resort so this remains usable.
    """
    if has_app_context():
        server_name = current_app.config.get("SERVER_NAME")
        if server_name:
            scheme = current_app.config.get("PREFERRED_URL_SCHEME", "http")
            return f"{scheme}://{server_name}"
    env_url = os.environ.get("PLUGIN_RUNNER_URL") or os.environ.get(
        "QHANA_PLUGIN_RUNNER_URL"
    )
    if env_url:
        return env_url.rstrip("/")
    return "http://localhost:5005"


def create_qasm_file_and_get_url(qasm_code: str) -> str:
    """
    Sends QASM code to the file creation plugin and retrieves the generated file URL.

    Args:
        qasm_code (str): The QASM code to be processed.

    Returns:
        str: The URL of the generated QASM file.

    Raises:
        ValueError: If any request fails or the response is invalid.
    """
    plugin_name = "filesMaker"
    version = "v0-0-1"
    base_url = _resolve_base_url()
    if has_app_context():
        try:
            api_url = url_for(f"{plugin_name}@{version}.ProcessView", _external=True)
        except Exception:
            api_url = f"{base_url}/plugins/{plugin_name}@{version}/process/"
    else:
        api_url = f"{base_url}/plugins/{plugin_name}@{version}/process/"

    max_retries = 10
    wait_seconds = 3.0
    qasm_payload = {"qasmCode": qasm_code}

    try:
        post_response = requests.post(api_url, data=qasm_payload, allow_redirects=False)
        post_response.raise_for_status()
    except requests.RequestException as error:
        raise ValueError(f"Failed to send POST request: {error}") from error

    redirect_url = post_response.headers.get("Location")
    if not redirect_url:
        raise ValueError("No redirect URL found in response headers.")

    # Ensure the redirect URL is absolute
    if not redirect_url.startswith("http"):
        redirect_url = f"{base_url}{redirect_url}"

    # Poll the redirect URL until a final result is available
    for _ in range(max_retries):
        try:
            get_response = requests.get(redirect_url)
            get_response.raise_for_status()
            result_data = get_response.json()
        except (requests.RequestException, json.JSONDecodeError) as error:
            raise ValueError(f"Error during polling: {error}") from error

        if result_data.get("status") != "PENDING":
            return result_data["outputs"][0]["href"]

        time.sleep(wait_seconds)

    raise TimeoutError("Polling exceeded maximum retries without a final result.")
