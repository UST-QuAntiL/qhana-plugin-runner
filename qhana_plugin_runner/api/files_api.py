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

"""Module containing the endpoints for the local filesystem file store."""

from http import HTTPStatus

import marshmallow as ma
from flask.globals import current_app
from flask.helpers import send_file
from flask.views import MethodView
from flask_smorest import abort

from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.db.models.tasks import TaskFile

FILES_API = SmorestBlueprint(
    "files-api",
    __name__,
    description="Api to download task result files.",
    url_prefix="/files",
)


class FileSecurityTagSchema(MaBaseSchema):
    file_id = ma.fields.String(
        required=True,
        allow_none=False,
        data_key="file-id",
        metadata={"description": "The security tag of the file."},
    )


@FILES_API.route("/<int:file_id>/")
class FileView(MethodView):
    """Download task result file stored in the local file-system store."""

    @FILES_API.arguments(FileSecurityTagSchema, location="query")
    @FILES_API.response(HTTPStatus.OK,)
    def get(self, query_data, file_id: int):
        """Get the task file information by file id."""
        security_tag = query_data["file_id"]  # prevent simple file id enumeration attacs
        task_file: TaskFile = TaskFile.get_by_id(file_id)
        if (
            not task_file
            or task_file.storage_provider != "local_filesystem"
            or task_file.security_tag != security_tag
        ):
            abort(HTTPStatus.NOT_FOUND, message="File not found.")
        if task_file.file_type == "temp-file":
            current_app.logger.warning(
                f"The temporary file {task_file.file_name} was exposed as a task result."
            )
        return send_file(
            task_file.file_storage_data,
            mimetype=task_file.mimetype,
            download_name=task_file.file_name,
        )
