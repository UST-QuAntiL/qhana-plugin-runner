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

"""Functions for opening files from external URLs."""

import mimetypes
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from re import Pattern
from typing import Iterator, Optional, Tuple
from urllib.parse import urlparse

from flask import Flask
from flask.globals import current_app
from pyrfc6266 import requests_response_to_filename, secure_filename
from requests import Session
from requests.models import Response
from werkzeug.http import parse_options_header

REQUEST_SESSION = Session()


def open_url(url: str, raise_on_error_status=True, **kwargs) -> Response:
    """Open an url with request.

    (see :py:meth:`~requests.Session.request` for parameters)

    It is best to use this function in a ``with``-statement to get autoclosing behaviour
    for the returned response. The returned response acts as a context manager.

    For streaming access set ``stream=True``.

    An appropriate exception is raised for an error status. To ignore an error status set ``raise_on_error_status=False``.
    """
    if current_app:
        # apply rewrite rules from the current app context in sequence
        app: Flask = current_app
        pattern: Pattern
        replacement: str
        for pattern, replacement in app.config.get("URL_REWRITE_RULES", []):
            url = pattern.sub(replacement, url)

    url_data = REQUEST_SESSION.get(url, **kwargs)
    if raise_on_error_status:
        url_data.raise_for_status()
    return url_data


@contextmanager
def open_url_as_file_like(
    url: str, raise_on_error_status=True, stream=True, **kwargs
) -> Iterator[Tuple[str, BytesIO, Optional[str]]]:
    """Open an url with requests to be used as a ile like object.

    This method should be used as a context manager, i.e., inside a with block,
    and returns a tuple (filename, file_like, content_type).

    This method uses :py:func:`open_url`.

    The ``file_like``object may be a ``urllib3.response.HTTPResponse`` object
    or another object with a ``read`` method depending on the used url adapter.

    Example:

    >>> with open_url_as_file_like(url) as (filename, file_like, _content_type):
    >>>     print(filename, file_like.read(128))
    """
    response = open_url(url, raise_on_error_status, stream=stream, **kwargs)

    response.raw.decode_content = True

    filename: str = "unknown"

    if hasattr(response.raw, "name"):
        # for file:// urls raw is directly the opened file
        filename = response.raw.name
    else:
        filename = secure_filename(requests_response_to_filename(response))

    content_type: Optional[str] = get_mimetype(response)

    try:
        yield filename, response.raw, content_type
    finally:
        response.close()


def get_mimetype(response: Response, default=None) -> Optional[str]:
    try:
        return response.headers["Content-Type"]
    except KeyError:
        matches = mimetypes.MimeTypes().guess_type(url=response.url)
        if matches:
            return matches[0]
    return default


def _retrieve_filename(response: Response):
    """
    Given an url response it returns the name of the file
    :param response: Response
    :return: str
    """
    url = response.url
    fname = None
    if "Content-Disposition" in response.headers.keys():
        for content_disp in parse_options_header(response.headers["Content-Disposition"]):
            if isinstance(content_disp, dict) and "filename" in content_disp:
                fname = content_disp["filename"]
                break
    if not fname:
        fname = Path(urlparse(url).path).name
    response.close()

    # Remove file type endings
    fname = Path(fname).stem

    return fname


def retrieve_filename(url_or_response: str | Response) -> str:
    """
    Given an url to a file or an url response, it returns the name of the file
    :param url_or_response: str | Response
    :return: str
    """
    if isinstance(url_or_response, str):  # url_or_response is an url
        with open_url(url_or_response, stream=True) as response:
            return _retrieve_filename(response)
    elif isinstance(url_or_response, Response):  # url_or_response is a response
        return _retrieve_filename(url_or_response)

    raise TypeError("Expected input to be str or request.Response.")
