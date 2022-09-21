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

from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, List

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response
from flask import redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE, post_load

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
    InputDataMetadata,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
    FileUrl,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
import json

_plugin_name = "one-hot encoding"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


ONEHOT_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="One-hot encoding plugin API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        entities_url: str,
        entities_metadata_url: str,
        taxonomies_zip_url: str,
        attributes: str,
    ):
        self.entities_url = entities_url
        self.entities_metadata_url = entities_metadata_url
        self.taxonomies_zip_url = taxonomies_zip_url
        self.attributes = attributes


class InputParametersSchema(FrontendFormBaseSchema):
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entities",
        data_content_types="application/json",
        metadata={
            "label": "Entities URL",
            "description": "URL to a file with entities.",
            "input_type": "text",
        },
    )
    entities_metadata_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="attribute-metadata",
        data_content_types="application/json",
        metadata={
            "label": "Entities Attribute Metadata URL",
            "description": "URL to a file with the attribute metadata for the entities.",
            "input_type": "text",
        },
    )
    taxonomies_zip_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="taxonomy",
        data_content_types="application/zip",
        metadata={
            "label": "Taxonomies URL",
            "description": "URL to zip file with taxonomies.",
            "input_type": "text",
        },
    )
    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Attributes",
            "description": "Attributes for which the similarity shall be computed.",
            "input_type": "textarea",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@ONEHOT_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @ONEHOT_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @ONEHOT_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """One-hot encoding endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="One-Hot Encoding",
            description=OneHot.instance.description,
            name=OneHot.instance.identifier,
            version=OneHot.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{ONEHOT_BLP.name}.CalcView"),
                ui_href=url_for(f"{ONEHOT_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entities",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesUrl"
                    ),
                    InputDataMetadata(
                        data_type="taxonomy",
                        content_type=["application/zip"],
                        required=True,
                        parameter="taxonomiesZipUrl"
                    ),
                    InputDataMetadata(
                        data_type="attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesMetadataUrl"
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity-points",
                        content_type=["application/csv"],
                        required=True,
                    )
                ],
            ),
            tags=OneHot.instance.tags,
        )


@ONEHOT_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the One-Hot encoding plugin."""

    @ONEHOT_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the ONEHOT plugin.",
    )
    @ONEHOT_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @ONEHOT_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @ONEHOT_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the one-hot encoding plugin.",
    )
    @ONEHOT_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @ONEHOT_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)

        return Response(
            render_template(
                "simple_template.html",
                name=OneHot.instance.name,
                version=OneHot.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{ONEHOT_BLP.name}.CalcView"),
            )
        )


@ONEHOT_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @ONEHOT_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @ONEHOT_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @ONEHOT_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=InputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = calculation_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class OneHot(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Converts Muse Data to One-Hot Encodings"
    tags = ["encoding", "one-hot"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return ONEHOT_BLP

    # def get_requirements(self) -> str:
    #     return "scikit-learn~=0.24.2"


TASK_LOGGER = get_task_logger(__name__)


def get_attribute_ref_target(entities_metadata_url: str, attributes: List[str]):
    result = dict()
    entities_metadata = open_url(entities_metadata_url).json()
    for attribute in attributes:
        for metadata in entities_metadata:
            # TASK_LOGGER.info(f"metadata[ID]: {metadata['ID']}")
            if metadata["ID"] == attribute:
                result[attribute] = metadata["refTarget"].split(":")[1][:-5]
    return result


def get_taxonomies_by_ref_target(attribute_ref_targets: dict, taxonomies_zip_url: str):
    ref_targets = set()
    for ref_target in attribute_ref_targets.values():
        ref_targets.add(ref_target)

    taxonomies = {}
    for zipped_file, file_name in get_files_from_zip_url(taxonomies_zip_url, mode="t"):
        if file_name[:-5] in ref_targets:
            taxonomy = json.load(zipped_file)
            taxonomies[file_name[:-5]] = taxonomy

    return taxonomies


def attribute_to_onehot(entity, attribute: str, attribute_ref_targets: dict, taxonomies) -> List[int]:
    taxonomy = taxonomies[attribute_ref_targets[attribute]]
    values = entity[attribute]
    sub_attributes = set()
    if type(values) == list:
        sub_attributes = set(values)
    else:
        sub_attributes.add(values)
    new_sub_a = set(sub_attributes.copy())
    while len(new_sub_a) != 0:
        old_sub_a = new_sub_a.copy()
        new_sub_a = set()
        for sub_a in old_sub_a:
            for relation in taxonomy['relations']:
                if relation['target'] == sub_a:
                    if relation['source'] != "":
                        new_sub_a.add(relation['source'])
        for new_a in new_sub_a:
            sub_attributes.add(new_a)

    vector = [0]*len(taxonomy['entities'])
    for i in range(len(vector)):
        if taxonomy['entities'][i] in sub_attributes:
            vector[i] = 1
    return vector


def entity_to_onehot(entity, attributes, attribute_ref_targets: dict, taxonomies) -> List[int]:
    vector = []
    for attribute in attributes:
        attribute_vector = attribute_to_onehot(entity, attribute, attribute_ref_targets, taxonomies)
        vector += attribute_vector
    return vector


def get_dim(attributes, attribute_ref_targets: dict, taxonomies):
    dim = 0
    for attribute in attributes:
        dim += len(taxonomies[attribute_ref_targets[attribute]]['entities'])
    return dim


def get_entity_dict(id, point):
    ent = {"ID": id, "href": ""}
    for d in range(len(point)):
        ent[f"dim{d}"] = point[d]
    return ent


def prepare_stream_output(entities, attributes, attribute_ref_targets, taxonomies):
    for entity in entities:
        onehot = entity_to_onehot(entity, attributes, attribute_ref_targets, taxonomies)
        yield get_entity_dict(entity["ID"], onehot)


@CELERY.task(name=f"{OneHot.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new one-hot encoding calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entities_url = input_params.entities_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entities_url='{entities_url}'"
    )
    entities_metadata_url = input_params.entities_metadata_url
    TASK_LOGGER.info(f"Loaded input parameters from db: entities_metadata_url='{entities_metadata_url}'")
    taxonomies_zip_url = input_params.taxonomies_zip_url
    TASK_LOGGER.info(f"Loaded input parameters from db: taxonomies_zip_url='{taxonomies_zip_url}'")
    attributes = input_params.attributes
    TASK_LOGGER.info(f"Loaded input parameters from db: entities_metadata_url='{attributes}'")

    # load data from file
    attributes = attributes.replace('\r', '').split('\n')
    attribute_ref_targets = get_attribute_ref_target(entities_metadata_url, attributes)
    taxonomies = get_taxonomies_by_ref_target(attribute_ref_targets, taxonomies_zip_url)

    entities = open_url(entities_url).json()
    dim = get_dim(attributes, attribute_ref_targets, taxonomies)
    csv_attributes = ["ID", "href"] + [f"dim{d}" for d in range(dim)]
    entity_points = prepare_stream_output(entities, attributes, attribute_ref_targets, taxonomies)

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entity_points, output, "text/csv", attributes=csv_attributes)
        STORE.persist_task_result(
            db_id,
            output,
            "one-hot-encoded_points.csv",
            "entity-points",
            "text/csv",
        )

    return "Result stored in file"
