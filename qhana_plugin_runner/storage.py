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

"""Module containing a file store interface with a implementation for the local file system."""

from pathlib import Path
from secrets import token_urlsafe
from shutil import copyfileobj
from typing import IO, BinaryIO, ClassVar, Dict, Optional, TextIO, Type, Union

from flask.app import Flask
from flask.helpers import url_for

from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile


class FileStoreInterface:
    """Base class defining the file store interface."""

    def persist_file(
        self, file_: Union[IO, str, bytes], target: Union[str, Path], mimetype: str
    ):
        """Persist a file to the file storage.

        Args:
            file_ (Union[IO, str, bytes]): the file like object to perist to the storage.
            target (Union[str, Path]): the file path to save the file on the storage including the file name.
            mimetype (str): the content type of the data.
        """
        raise NotImplementedError()

    def persist_task_result(
        self,
        task_db_id: int,
        file_: Union[IO, str, bytes],
        file_name: str,
        file_type: str,
        mimetype: str,
        commit: bool = True,
    ) -> TaskFile:
        """Perist a task result file and store the file information in the database.

        Args:
            task_db_id (int): the id of the task in the database
            file_ (Union[IO, str, bytes]): the file object to persist
            file_name (Path): the file name of the result file
            file_type (str): the file type tag
            mimetype (str): the mime type of the file (not optional for result files!)
            commit (bool): if true commits the current DB transaction. Defaults to True.

        Raises:
            KeyError: if the task could not be found in the database

        Returns:
            TaskFile: the file information stored in the database
        """
        raise NotImplementedError()

    def persist_task_temp_file(
        self,
        task_db_id: int,
        file_: Union[IO, str, bytes],
        file_name: str,
        mimetype: Optional[str] = None,
        commit: bool = True,
    ) -> TaskFile:
        """Perist a temporary task file and store the file information in the database.

        Temporary files should only be used during the execution of the task.
        All temporary files have the file type tag ``"temp-file"``.

        Args:
            task_db_id (int): the id of the task in the database
            file_ (Union[IO, str, bytes]): the file object to persist
            file_name (str): the file name of the result file
            mimetype (Optional[str]): the mime type of the file. Defaults to None.
            commit (bool): if true commits the current DB transaction. Defaults to True.

        Returns:
            TaskFile: the file information stored in the database
        """
        raise NotImplementedError()

    def get_file_url(self, file_storage_data: str, external: bool = True) -> str:
        """Get a URL to the stored file.

        If ``external`` is ``False`` the file store implementation may return an
        internal URL that must work with :py:func:`~qhana_plugin_runner.requests.open_url`.

        If ``external`` is ``True`` the URL must be accessible from outside this
        microservice (e.g. through an API endpoint of this microservice) without
        authorization for at least 2 hours.

        Args:
            file_storage_data (str): the file metadata as defined by the file store or as stored in :py:attr:`~qhana_plugin_runner.db.models.tasks.TaskFile.file_storage_data`
            external (bool, optional): if the URL should be accessible from outside (for downloading the file). Defaults to True.

        Returns:
            str: the URL to the file
        """
        raise NotImplementedError()

    def get_task_file_url(self, file_info: TaskFile, external: bool = True) -> str:
        """Get an URL for a TaskFile object.

        See :py:meth:`~qhana_plugin_runner.storage.FileStoreRegistry.get_file_url`

        Args:
            file_info (TaskFile): the information of the task file

        Returns:
            str: the URL to the file
        """
        raise NotImplementedError()


class FileStore(FileStoreInterface):
    """Interface class for file store implementations."""

    __store_classes: ClassVar[Dict[str, Type["FileStore"]]] = {}

    name: ClassVar[str]  # populated by the FileStore instance for loaded implementations

    def __init_subclass__(cls, name: str, **kwargs) -> None:
        """Register file store implementation classes by their name."""
        cls.name = name
        FileStore.__store_classes[name] = cls

    def __init__(self, app: Optional[Flask] = None) -> None:
        super().__init__()
        self.app: Optional[Flask] = None
        if app:
            self.init_app(app)

    @staticmethod
    def _get_registered_store_classes():
        """Get the dict containing the registered FileStore implementation classes."""
        return FileStore.__store_classes

    def init_app(self, app: Flask):
        """Init the file store with the Flask app to get access to the flask config."""
        self.app = app

    def prepare_path(self, path: Union[str, Path]) -> Path:
        """Prepare a file path before saving a file.

        The returned path object is always a relative path.
        If the path is not relative a path relative to the root of that path is returned.

        Mitigates simple path traversal attacs.

        Args:
            path (Union[str, Path]): The file path to prepare

        Raises:
            ValueError: If the path contains potentially dangerous parts (e.g. ``./../file-txt``)

        Returns:
            Path: A relative path
        """
        if isinstance(path, str):
            path = Path(path)

        if ".." in path.parts:
            # prevent easy path traversal attacs
            raise ValueError("Paths may not contain parent folder parts ('..').")

        if path.is_absolute():
            path = path.relative_to(path.root)
        return path

    def _get_file_identifier(self, target: Union[Path, str]) -> str:
        """Get the full file identifier (e.g. the absolute path or a UIR/URL to the file)."""
        return str(target)

    def _persist_task_file(
        self,
        task_db_id: int,
        file_: Union[IO, str, bytes],
        target: Union[Path, str],
        file_name: str,
        file_type: str,
        mimetype: Optional[str] = None,
        commit: bool = True,
    ) -> TaskFile:
        """Perist a task file and store the file information in the database.

        Args:
            task_db_id (int): the id of the task in the database
            file_ (Union[IO, str, bytes]): the file object to persist
            target (Union[Path, str]): the target path or filename
            file_name (str): the file name
            file_type (str): the file type tag
            mimetype (Optional[str]): the mime type of the file
            commit (bool): if true commits the current DB transaction. Defaults to True.

        Raises:
            KeyError: if the task could not be found in the database

        Returns:
            TaskFile: the file information stored in the database
        """
        task = ProcessingTask.get_by_id(task_db_id)
        if not task:
            raise KeyError(f"No task with database id {task_db_id} found!")
        if not mimetype:
            mimetype = "application/octet-stream"
        self.persist_file(file_, target, mimetype)
        file_info = TaskFile(
            task=task,
            security_tag=token_urlsafe(32),
            storage_provider=self.name,
            file_name=file_name,
            file_storage_data=self._get_file_identifier(target),
            file_type=file_type,
            mimetype=mimetype,
        )
        file_info.save(commit)
        return file_info

    def persist_task_result(
        self,
        task_db_id: int,
        file_: Union[IO, str, bytes],
        file_name: str,
        file_type: str,
        mimetype: str,
        commit: bool = True,
    ) -> TaskFile:
        target = Path(f"task_{task_db_id}/out") / Path(file_name)
        return self._persist_task_file(
            task_db_id, file_, target, file_name, file_type, mimetype, commit
        )

    def persist_task_temp_file(
        self,
        task_db_id: int,
        file_: Union[IO, str, bytes],
        file_name: str,
        mimetype: Optional[str] = None,
        commit: bool = True,
    ) -> TaskFile:
        target = Path(f"task_{task_db_id}/tmp") / Path(file_name)
        return self._persist_task_file(
            task_db_id,
            file_,
            target,
            file_name,
            file_type="temp-file",
            mimetype=mimetype,
            commit=commit,
        )

    def get_task_file_url(self, file_info: TaskFile, external: bool = True) -> str:
        return self.get_file_url(file_info.file_storage_data, external=external)


class LocalFileStore(FileStore, name="local_filesystem"):
    """A file store implementation using the local file system."""

    def __init__(self, app: Flask) -> None:
        super().__init__(app=app)
        self._root_path: Optional[Path] = None

    def _get_storage_root(self) -> Path:
        """Get the root path where to store the files at.

        The root path can be configured with ``"FILE_STORE_ROOT_PATH"`` in the app config.

        A relative path will be interpreted relative to the app inctance folder.
        """
        if self._root_path:
            return self._root_path
        assert self.app is not None
        settings_path = Path(self.app.config.get("FILE_STORE_ROOT_PATH", "files"))
        if settings_path.is_absolute():
            self._root_path = settings_path
        else:
            self._root_path = Path(self.app.instance_path) / settings_path
        return self._root_path

    def _get_file_identifier(self, target: Path):
        """Get the full path of the file on disc."""
        return str(self._get_storage_root() / target)

    def persist_raw_data(
        self, data: Union[str, bytes], target: Union[str, Path], mimetype: str
    ):
        pass
        target_path = self._get_storage_root() / self.prepare_path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if isinstance(data, bytes) else "w"
        with target_path.open(mode=mode) as target_file:
            target_file.write(data)

    def persist_file(
        self, file_: Union[IO, str, bytes], target: Union[str, Path], mimetype: str
    ):
        mode: str  # mode to open target file with
        if isinstance(file_, (str, bytes)):
            self.persist_raw_data(file_, target, mimetype)
            return
        elif isinstance(file_, TextIO):
            mode = "w"
        elif isinstance(file_, BinaryIO):
            mode = "wb"
        elif hasattr(file_, "mode"):
            # try to guess if text or binary mode was used
            mode = "wb" if ("b" in file_.mode) else "w"
        else:
            raise ValueError("Cannot determine mode of file object!")
        if hasattr(file_, "seek") and callable(file_.seek):
            try:  # seek to beginning of a file (useful for in memory temp files that were just written)
                file_.seek(0)
            except Exception:
                pass  # assume the file object does not support seek
        target_path = self._get_storage_root() / self.prepare_path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open(mode=mode) as target_file:
            copyfileobj(file_, target_file)

    def get_file_url(self, file_storage_data: str, external: bool = True) -> str:
        if not external:
            # return an internal file url
            return "file://" + file_storage_data
        raise NotImplementedError()  # TODO implement endpoint to download files by file path

    def get_task_file_url(self, file_info: TaskFile, external: bool = True) -> str:
        if not external:
            return super().get_task_file_url(file_info, external=external)
        return url_for(
            "files-api.FileView",
            file_id=file_info.id,
            **{"file-id": file_info.security_tag},
            _external=True,
        )


class UrlFileStore(FileStore, name="url_file_store"):
    """A file store implementation using url references."""

    def __init__(self, app: Flask) -> None:
        super().__init__(app=app)

    def persist_file(
        self, file_: Union[IO, str, bytes], target: Union[str, Path], mimetype: str
    ):
        if not isinstance(file_, str):
            raise ValueError(
                "UrlFileStorage can only persist an URL reference passed as a string. E.g. file_='http://...'"
            )

    def _persist_task_file(
        self,
        task_db_id: int,
        file_: Union[IO, str, bytes],
        target: Union[Path, str],
        file_name: str,
        file_type: str,
        mimetype: Optional[str] = None,
        commit: bool = True,
    ) -> TaskFile:
        task = ProcessingTask.get_by_id(task_db_id)
        if not task:
            raise KeyError(f"No task with database id {task_db_id} found!")
        if not mimetype:
            mimetype = "application/octet-stream"
        self.persist_file(file_, target, mimetype)
        assert isinstance(file_, str)
        file_info = TaskFile(
            task=task,
            security_tag=token_urlsafe(32),
            storage_provider=self.name,
            file_name=file_name,
            file_storage_data=file_,
            file_type=file_type,
            mimetype=mimetype,
        )
        file_info.save(commit)
        return file_info

    def get_file_url(self, file_storage_data: str, external: bool = True) -> str:
        return file_storage_data


class FileStoreRegistry(FileStoreInterface):
    """Class acting as a registry for loaded file stores. Forwards calls to the default file store."""

    def __init__(self, app: Optional[Flask] = None) -> None:
        super().__init__()
        self.app: Optional[Flask] = None
        self._default_store: Optional[str] = None
        self._stores: Dict[str, FileStore] = {}
        if app:
            self.init_app(app)

    def __getitem__(self, item: str):
        # allow getting specific file stores with file_store[<name>]
        return self._stores[item]

    def _load_registered_stores(self):
        """Load and instantiate all registered file store implementations.

        This method must only be called on the generic ``FileStore`` instance
        and not on file store implementations!

        The created file store implementation objects will have this ``FileStore``
        instance set as their :py:attr:`~qhana_plugin_runner.storage.FileStore._store_registry`.
        """
        for name, cls in FileStore._get_registered_store_classes().items():
            try:
                self._stores[name] = cls(app=self.app)
            except Exception as e:
                self.app.logger.warning(
                    f"Could not load storage provider {name} because of the follwoing exception.",
                    exc_info=True,
                )
        self._set_default_store_from_config()

    def init_app(self, app: Flask):
        """Init the file store with the Flask app to get access to the flask config."""
        self.app = app
        self._set_default_store_from_config()
        failed_stores = set()
        for key, store in self._stores.items():
            try:
                store.init_app(app)
            except:
                app.logger.warning(
                    f"Failed to initialize File Store '{key}'. The file store will be removed.",
                    exc_info=True,
                )
                failed_stores.add(key)
        for key in failed_stores:
            del self._stores[key]

    def _set_default_store_from_config(self):
        """Read and set the default store implementation to use from flask config."""
        if self.app:
            self._default_store = self.app.config.get("DEFAULT_FILE_STORE")

    def persist_file(
        self,
        file_: Union[IO, str, bytes],
        target: Union[str, Path],
        mimetype: str = "application/octet-stream",
        storage_provider: Optional[str] = None,
    ):
        if storage_provider is None:
            storage_provider = self._default_store
        if storage_provider is None:
            raise NotImplementedError()
        self._stores[storage_provider].persist_file(file_, target, mimetype)

    def persist_task_result(
        self,
        task_db_id: int,
        file_: Union[IO, str, bytes],
        file_name: str,
        file_type: str,
        mimetype: str,
        commit: bool = True,
        storage_provider: Optional[str] = None,
    ) -> TaskFile:
        if storage_provider is None:
            storage_provider = self._default_store
        if storage_provider is None:
            raise NotImplementedError()
        return self._stores[storage_provider].persist_task_result(
            task_db_id, file_, file_name, file_type, mimetype, commit=commit
        )

    def persist_task_temp_file(
        self,
        task_db_id: int,
        file_: Union[IO, str, bytes],
        file_name: str,
        mimetype: Optional[str] = None,
        commit: bool = True,
        storage_provider: Optional[str] = None,
    ) -> TaskFile:
        if storage_provider is None:
            storage_provider = self._default_store
        if storage_provider is None:
            raise NotImplementedError()
        return self._stores[storage_provider].persist_task_temp_file(
            task_db_id, file_, file_name, mimetype, commit=commit
        )

    def get_file_url(
        self,
        file_storage_data: str,
        external: bool = True,
        storage_provider: Optional[str] = None,
    ) -> str:
        if storage_provider is None:
            storage_provider = self._default_store
        if storage_provider is None:
            raise NotImplementedError()
        return self._stores[storage_provider].get_file_url(file_storage_data, external)

    def get_task_file_url(self, file_info: TaskFile, external: bool = True) -> str:
        storage_provider = (
            file_info.storage_provider
            if file_info.storage_provider
            else self._default_store
        )
        if storage_provider is None:
            raise NotImplementedError()
        return self._stores[storage_provider].get_task_file_url(file_info, external)


# The file store registry that should be imported and used
STORE: FileStoreRegistry = FileStoreRegistry()


def register_file_store(app: Flask):
    """Register the file store instance with the flask app."""
    STORE._load_registered_stores()
    STORE.init_app(app)
