# Copyright 2024 QHAna plugin runner contributors.
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


from http import HTTPStatus
from typing import Any, List, Optional, Tuple
from urllib.parse import urljoin
from base64 import urlsafe_b64decode
from json import loads
from time import time

import requests


class Muse4MusicError(Exception):
    """Base class for all Muse4Music client errors."""


def base64url_decode(input_: str) -> bytes:
    input_bytes = input_.encode()

    rem = len(input_bytes) % 4

    if rem > 0:
        input_bytes += b"=" * (4 - rem)

    return urlsafe_b64decode(input_bytes)


class Muse4MusicClient:

    _timeout: float
    _base_url: str
    _auth_token: Optional[str] = None
    _refresh_token: Optional[str] = None

    def __init__(self, base_url: str, timeout: float = 3) -> None:
        self._base_url = base_url
        self._timeout = timeout

    def login(self, username: str, password: str) -> Tuple[str, str]:
        url = urljoin(self._base_url, "./user-api/auth/login/")
        response = requests.post(
            url, json={"username": username, "password": password}, timeout=self._timeout
        )
        response.raise_for_status()
        match response.json():
            case {"access_token": str(auth_token), "refresh_token": str(refresh_token)}:
                self._auth_token = auth_token
                self._refresh_token = refresh_token
                return auth_token, refresh_token
        raise ValueError("Received incorrect response to login request.", response)

    def _ensure_logged_in(self):
        if self._auth_token is None or self._refresh_token is None:
            raise Muse4MusicError("Client is not logged in to MUSE4Music!")
        auth_data = loads(base64url_decode(self._auth_token.split(".")[1]))
        now = time()
        future = now + 30
        if (auth_data.get("iat", now) > now) or (auth_data.get("exp", future) < future):
            self.refresh_tokens()

    def _get_by_url(self, url):
        self._ensure_logged_in()
        url = urljoin(self._base_url, url)
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def test_login(self) -> Optional[str]:
        self._ensure_logged_in()
        url = urljoin(self._base_url, "./user-api/auth/check/")
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json().get("username")

    def refresh_tokens(self) -> bool:
        if self._refresh_token is None:
            raise Muse4MusicError("Client is not logged in to MUSE4Music!")
        url = urljoin(self._base_url, "./user-api/auth/refresh/")
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {self._refresh_token}"},
            timeout=self._timeout,
        )
        if response.status_code in (HTTPStatus.FORBIDDEN, HTTPStatus.UNAUTHORIZED):
            return False
        response.raise_for_status()
        match response.json():
            case {"access_token": str(auth_token)}:
                self._auth_token = auth_token
                return True
        return False

    def get_taxonomies(self) -> List[str]:
        self._ensure_logged_in()
        url = urljoin(self._base_url, "./api/taxonomies/")
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return [t["_links"]["self"]["href"] for t in response.json().get("taxonomies")]

    def get_taxonomy(self, url):
        return self._get_by_url(url)

    def get_people(self) -> List[Any]:
        self._ensure_logged_in()
        url = urljoin(self._base_url, "./api/persons/")
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_person(self, url):
        return self._get_by_url(url)

    def get_opuses(self) -> List[Any]:
        self._ensure_logged_in()
        url = urljoin(self._base_url, "./api/opuses/")
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_opus(self, url):
        return self._get_by_url(url)

    def get_parts(self) -> List[Any]:
        self._ensure_logged_in()
        url = urljoin(self._base_url, "./api/parts/")
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_part(self, url):
        return self._get_by_url(url)

    def get_subparts(self) -> List[Any]:
        self._ensure_logged_in()
        url = urljoin(self._base_url, "./api/subparts/")
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_subpart(self, url):
        return self._get_by_url(url)

    def get_voices(self, url) -> List[Any]:
        self._ensure_logged_in()
        subpart_url = urljoin(self._base_url, url)
        url = urljoin(subpart_url, "./voices/")
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._auth_token}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_voice(self, url):
        return self._get_by_url(url)
