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

from pathlib import Path
from shutil import copyfileobj
from typing import IO, BinaryIO, ClassVar, Dict, Optional, TextIO, Type, Union

from flask.app import Flask

from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile

# TODO add documentation


class FileStore:

    __store_classes: ClassVar[Dict[str, Type["FileStore"]]] = {}

    _name: ClassVar[str]

    def __init_subclass__(cls, name: str, **kwargs) -> None:
        cls._name = name
        FileStore.__store_classes[name] = cls

    def __init__(self, app: Flask = None) -> None:
        super().__init__()
        self.app: Optional[Flask] = None
        self._default_store: Optional[str] = None
        self._stores: Dict[str, FileStore] = {}
        if app:
            self.init_app(app)

    def load_registered_stores(self):
        for name, cls in FileStore.__store_classes.items():
            self._stores[name] = cls(app=self.app)
        self._set_default_store_from_config()

    def init_app(self, app: Flask):
        self.app = app
        self._set_default_store_from_config()
        for store in self._stores.values():
            store.init_app(app)

    def _set_default_store_from_config(self):
        if self.app:
            self._default_store = self.app.config.get("DEFAULT_FILE_STORE")

    @property
    def name(self):
        try:
            if self._name:
                return self._name
        except AttributeError:
            pass  # all stores (except FileStore) should have their _name attribute set.
        # assume default store was used
        return self._default_store

    def prepare_path(self, path: Union[str, Path]) -> Path:
        if isinstance(path, str):
            path = Path(path)

        if ".." in path.parts:
            # prevent easy path traversal attacs
            raise ValueError("Paths may not contain parent folder parts ('..').")

        if path.is_absolute():
            path = path.relative_to(path.root)
        return path

    def _get_file_storage_data(self, target: Path) -> str:
        assert type(self) == FileStore, "Do not call this method with super()!"
        if self._default_store is None:
            raise NotImplementedError()
        return self._stores[self._default_store]._get_file_storage_data(target)

    def persist_file(self, file_: IO, target: Union[str, Path]):
        assert type(self) == FileStore, "Do not call this method with super()!"
        if self._default_store is None:
            raise NotImplementedError()
        return self._stores[self._default_store].persist_file(file_, target)

    def persist_task_result(
        self, task_db_id: int, file_: IO, file_name: str, file_type: str, mimetype: str
    ):
        task = ProcessingTask.get_by_id(task_db_id)
        if not task:
            raise KeyError(f"No task with database id {task_db_id} found!")
        target = Path(f"task_{task_db_id}/out") / Path(file_name)
        self.persist_file(file_, target)
        TaskFile(
            task=task,
            storage_provider=self.name,
            file_name=file_name,
            file_storage_data=self._get_file_storage_data(target),
            file_type=file_type,
            mimetype=mimetype,
        ).save(True)

    def persist_task_temp_file(
        self, task_db_id: int, file_: IO, file_name: str, mimetype: Optional[str] = None
    ):
        task = ProcessingTask.get_by_id(task_db_id)
        if not task:
            raise KeyError(f"No task with database id {task_db_id} found!")
        target = Path(f"task_{task_db_id}/tmp") / Path(file_name)
        self.persist_file(file_, target)
        TaskFile(
            task=task,
            storage_provider=self.name,
            file_name=file_name,
            file_storage_data=self._get_file_storage_data(target),
            file_type="temp-file",
            mimetype=mimetype,
        ).save(True)


class LocalFileStore(FileStore, name="local_filesystem"):
    def __init__(self, app: Flask) -> None:
        super().__init__(app=app)

    def _get_storage_root(self) -> Path:
        assert self.app is not None
        return Path(self.app.instance_path) / Path("files")

    def _get_file_storage_data(self, target: Path) -> str:
        return str(self._get_storage_root() / target)

    def persist_file(self, file_: IO, target: Union[str, Path]):
        mode: str  # mode to open target file with
        if isinstance(file_, TextIO):
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


STORE: FileStore = FileStore()


def register_file_store(app: Flask):
    STORE.load_registered_stores()
    STORE.init_app(app)
