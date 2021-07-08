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

"""Adapters to load ``file://`` and ``data:`` URLs with :py:mod:`requests`"""

from base64 import b64decode
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import Container, Mapping, Optional, Text, Tuple, Union
from urllib.parse import unquote_to_bytes, urlparse

from requests import Session
from requests.adapters import BaseAdapter
from requests.models import PreparedRequest, Response


class FileAdapter(BaseAdapter):
    """Adapter to load ``file://`` URLs."""

    def send(self, request: PreparedRequest, stream: bool, **kwargs) -> Response:
        if request.method not in ("HEAD", "GET"):
            raise ValueError(f"Ussupported request method ({request.method}).")

        if request.url is None:
            raise ValueError("Ussupported request without url!")

        url = urlparse(request.url)
        if url.netloc and url.netloc != "localhost":
            raise ValueError(f"Only localhost is supported for file:// URLs!")

        path = url.path
        if isinstance(path, bytes):
            path = path.decode()

        file_path = Path(path)

        resp = Response()
        resp.url = request.url

        if file_path.is_dir():
            resp.status_code = HTTPStatus.FORBIDDEN

        if not file_path.exists():
            resp.status_code = HTTPStatus.NOT_FOUND

        if resp.status_code is None:  # if no error
            try:
                file_object = file_path.open(mode="rb")  # read binary
                resp.raw = file_object

                resp.headers["Content-Length"] = str(file_path.stat().st_size)
                resp.status_code = HTTPStatus.OK

                # TODO set mimetype in response?
            except IOError:
                resp.status_code = HTTPStatus.INTERNAL_SERVER_ERROR

        if resp.status_code != HTTPStatus.OK:
            body = resp.status_code.description.encode()  # only works with HTTPStatus instances!
            resp.raw = BytesIO(body)
            resp.headers["Content-Length"] = str(len(body))

        return resp


class DataAdapter(BaseAdapter):
    """Adapter to load ``data:`` URLs."""

    def send(self, request: PreparedRequest, stream: bool, **kwargs) -> Response:
        if request.method not in ("HEAD", "GET"):
            raise ValueError(f"Ussupported request method ({request.method}).")

        if request.url is None:
            raise ValueError("Ussupported request without url!")

        data_url = request.url
        if not data_url.startswith("data:"):
            raise ValueError(f"Ussupported data url {data_url}.")

        prefix, data = data_url[5:].split(",", maxsplit=1)

        is_base64 = prefix.endswith(";base64")
        if is_base64:
            prefix = prefix[:-7]

        mimetype = "text/plain"
        charset = "US-ASCII"

        if ";charset=" in prefix:
            mimetype, charset = prefix.split(";charset=")
        elif prefix:
            mimetype = prefix

        # TODO set mimetype and charset in response?

        resp = Response()
        resp.url = request.url

        if is_base64:
            body = b64decode(data)
        else:
            body = unquote_to_bytes(data)
        resp.status_code = HTTPStatus.OK
        resp.raw = BytesIO(body)
        resp.headers["Content-Length"] = str(len(body))

        return resp


def register_additional_schemas(session: Session):
    """Register adapters for the additional schemas in this module with a requests session."""
    session.mount("file://", FileAdapter())
    session.mount("data:", DataAdapter())
