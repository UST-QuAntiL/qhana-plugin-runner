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

from requests import Session
from requests.models import Response

REQUEST_SESSION = Session()


def open_url(url: str, **kwargs) -> Response:
    """Open an url with request.

    (see :py:meth:`~requests.Session.request` for parameters)

    It is best to use this function in a ``with``-statement to get autoclosing behaviour
    for the returned response. The returned response acts as a context manager.

    For streaming access set ``stream=True``.
    """
    return REQUEST_SESSION.get(url, **kwargs)
