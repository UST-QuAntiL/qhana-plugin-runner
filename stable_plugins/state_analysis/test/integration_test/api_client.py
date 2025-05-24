import json
import time

import pytest
import requests


class APIClient:
    def __init__(self, base_url: str, plugin_name: str, version: str):
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
            # Step 1: POST request to initiate the process
            response = requests.post(self.api_url, data=data, allow_redirects=False)
            response.raise_for_status()

            # Step 2: Retrieve the redirect URL from response headers
            redirect_url = response.headers.get("Location")
            if not redirect_url:
                raise ValueError("No redirect URL found in the response headers.")

            if not redirect_url.startswith("http"):
                redirect_url = self.base_url + redirect_url

            # Step 3: GET request to fetch the result, ggf. mehrfach pollen
            for attempt in range(max_retries):
                redirect_res = requests.get(redirect_url)
                redirect_res.raise_for_status()

                result_data = redirect_res.json()

                # Wenn kein 'status' drinsteht oder wenn status != 'PENDING', direkt zurückgeben
                if result_data.get("status") != "PENDING":
                    return result_data

                # Ansonsten warten und nochmal probieren
                time.sleep(wait_seconds)

            # Falls nach allen Retries noch immer 'PENDING', Fehler werfen
            raise ValueError(f"Result is still 'PENDING' after {max_retries} retries.")

        except requests.RequestException as req_err:
            raise ValueError(f"Request error: {req_err}") from req_err
        except (ValueError, json.JSONDecodeError) as json_err:
            raise ValueError(f"JSON error: {json_err}") from json_err
        except Exception as exc:
            raise ValueError(f"Unexpected error: {exc}") from exc
