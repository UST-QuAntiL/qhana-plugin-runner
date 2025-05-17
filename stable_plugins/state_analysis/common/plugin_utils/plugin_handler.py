import json
import time

import requests

# Hardcoded API endpoint; may need to be adjusted.
base_url = "http://localhost:5005"


def create_qasm_file_and_get_url(qasm_code: str) -> str:
    """
    Sends a QASM code to the file creation plugin and retrieves the generated file URL.

    Args:
        qasm_code (str): The QASM code to be processed.

    Returns:
        str: The URL of the generated QASM file.

    Raises:
        ValueError: If any request fails or the response is invalid.
    """
    plugin_name = "filesMaker"
    version = "v0-0-1"
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
