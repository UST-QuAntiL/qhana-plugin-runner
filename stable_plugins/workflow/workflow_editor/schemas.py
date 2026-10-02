# Copyright 2025 QHAna plugin runner contributors.
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

import marshmallow as ma
from marshmallow.validate import OneOf

from qhana_plugin_runner.api.util import MaBaseSchema


class WorkflowSchema(MaBaseSchema):
    id = ma.fields.String()
    version = ma.fields.String()
    name = ma.fields.String()
    date = ma.fields.AwareDateTime()
    autosave = ma.fields.Bool(
        required=False, missing=False, description="If this save was an autosave."
    )
    workflow_id = ma.fields.String()


class WorkflowSaveParamsSchema(MaBaseSchema):
    autosave = ma.fields.Bool(
        required=False, missing=False, description="Set this to true for autosaves."
    )
    deploy = ma.fields.String(
        required=False,
        missing="",
        validate=OneOf(choices=("", "workflow", "plugin", "ui-template")),
        description=(
            "Set to 'plugin' to save and deploy workflow as a QHAna plugin. "
            "Set to 'workflow' to save and deploy workflow to camunda only. "
            "Set to 'ui-template' to turn the wokflow into a UI Templatefor QHAna."
        ),
    )
