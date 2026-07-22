import mimetypes
from http import HTTPStatus
from io import BytesIO, TextIOWrapper
from typing import IO, Any, Optional, Text, Tuple, Generator, Union
from zipfile import ZipFile

from requests.models import Response

from qhana_plugin_runner.requests import open_url


def get_files_from_zip_url(
    url: str, mode="t"
) -> Generator[Tuple[Union[IO[bytes], IO[Text]], str], Any, None]:
    with open_url(url) as taxonomy_data:
        zip_bytes = taxonomy_data.content
        # SpooledTemporaryFile cannot be used here because of https://bugs.python.org/issue26175
        tmp_buffer = BytesIO(zip_bytes)
        zip_file = ZipFile(tmp_buffer)

        for file_name in zip_file.namelist():
            with zip_file.open(file_name) as zipped_file:
                if "b" in mode:
                    yield zipped_file, file_name
                else:
                    yield TextIOWrapper(zipped_file), file_name


def get_file_responses_from_zip(
    zip_bytes: bytes, content_type: Optional[str] = None
) -> Generator[Response, Any, None]:
    """Yield a ``Response`` for every file in an in-memory zip.

    Each member is wrapped in a :py:class:`~requests.Response` so it can be
    consumed by the same helpers used for remote files (e.g.
    :py:func:`~qhana_plugin_runner.plugin_utils.entity_marshalling.load_entities`).
    Directory entries are skipped and members are yielded in sorted name order.

    The ``Content-Type`` header of each response is guessed from the member's
    file name; pass ``content_type`` to override this for all members.

    Args:
        zip_bytes: the raw bytes of the zip archive.
        content_type: optional content type to set on every response instead of
            guessing it from the file name.
    """
    with ZipFile(BytesIO(zip_bytes)) as zip_file:
        for file_name in sorted(zip_file.namelist()):
            if file_name.endswith("/"):
                continue  # skip directory entries

            response = Response()
            response.url = file_name
            response.status_code = HTTPStatus.OK
            response.encoding = "utf-8"
            response.raw = BytesIO(zip_file.read(file_name))

            mimetype = content_type or mimetypes.MimeTypes().guess_type(file_name)[0]
            if mimetype:
                response.headers["Content-Type"] = mimetype

            yield response
