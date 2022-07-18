# Copyright 2022 QHAna plugin runner contributors.
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

from collections.abc import Sized
from datetime import timedelta
from json import loads
from os import environ
from pathlib import Path
from typing import IO, BinaryIO, Optional, TextIO, Union, cast

from flask import Flask

from qhana_plugin_runner.db.models.tasks import TaskFile
from qhana_plugin_runner.storage import FileStore
from qhana_plugin_runner.util.plugins import QHAnaPluginBase

__version__ = "v0.1.0"


class HelloWorld(QHAnaPluginBase):
    """A plugin providing a minio storage provider."""

    name = "minio-storage"
    version = __version__
    description = "minio-storage"
    tags = []

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_requirements(self) -> str:
        return "minio"


import minio


class TextFileWrapper(BinaryIO):
    """A minimal wrapper around string io to produce bytes output."""

    def __init__(self, file_: TextIO):
        self._file = file_

    def read(self, *args):
        return self._file.read(*args).encode()


class MinioStore(FileStore, name="minio"):
    """A file store implementation using minio backend.

    Set the config option `DEFAULT_FILE_STORE` to "minio" to set this as the
    default file store.

    The minio client can be configured with the `MINIO_CLIENT` config key.
    The configuration is a mapping that is directly passed to `minio.Minio()`.
    The client configuration can also be set via the environment variable
    `MINIO_CLIENT`. The config needs to be encoded as a json object.

    All files will be stored in a single bucket. The bucket and all other
    settings of the storage provider use the `MINIO` config key. The default
    bucket can be configured by the setting `MINIO.bucket="custom-bucket"`.
    The default bucket can also be set via the `MINIO_BUCKET` environment variable.
    """

    def __init__(self, app: Flask) -> None:
        super().__init__(app=app)
        self._client: Optional[minio.Minio] = None
        self._minio_bucket: Optional[str] = None

    def init_app(self, app: Flask):
        """Init the file store with the Flask app to get access to the flask config."""
        super().init_app(app)
        minio_config = app.config.get("MINIO_CLIENT", {})
        env_config = environ.get("MINIO_CLIENT", None)
        if env_config:
            minio_config = loads(env_config)
        if not minio_config:
            raise ValueError("No configuration for the minio file store found.")
        self._minio_bucket = app.config.get("MINIO", {}).get("bucket", "experiment-data")
        self._minio_bucket = environ.get("MINIO_BUCKET", self._minio_bucket)
        self._client = minio.Minio(**minio_config)
        if not self._client.bucket_exists(self._minio_bucket):
            self._client.make_bucket(self._minio_bucket)

    def persist_file(
        self,
        file_: IO,
        target: Union[str, Path],
        mimetype: str = "application/octet-stream",
    ):
        is_text = isinstance(file_, TextIO) or (
            hasattr(file_, "mode") and "b" not in file_.mode
        )

        if hasattr(file_, "seek") and callable(file_.seek):
            try:  # seek to beginning of a file (useful for in memory temp files that were just written)
                file_.seek(0)
            except Exception:
                pass  # assume the file object does not support seek

        length = len(file_) if isinstance(file_, Sized) else -1

        if self._client is None:
            raise ValueError("Client not configured!")

        if is_text:
            # wrap text io objects to produce bytes on read
            file_ = TextFileWrapper(cast(TextIO, file_))

        extra_args = {}

        if length < 0:
            # part size >5MiB must be set if the size is unknown
            extra_args["part_size"] = 2 ** 25

        self._client.put_object(
            self._minio_bucket,
            str(target),
            file_,
            length=length,
            content_type=mimetype,
            **extra_args,
        )

    def get_file_url(self, file_storage_data: str, external: bool = True) -> str:
        if self._client is None:
            raise ValueError("Client not configured!")

        return self._client.get_presigned_url(
            "GET",
            self._minio_bucket,
            file_storage_data,
            expires=timedelta(days=(3 if external else 7)),
        )

    def get_task_file_url(self, file_info: TaskFile, external: bool = True) -> str:
        return super().get_task_file_url(file_info, external=external)
