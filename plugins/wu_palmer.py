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
import json
from http import HTTPStatus
from io import StringIO
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, List
from zipfile import ZipFile

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response
from flask import redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

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
    load_entities,
)
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "wu-palmer"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


WU_PALMER_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Wu Palmer plugin API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


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
    wu_palmer_cache_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="wu-palmer-cache",
        data_content_types="application/zip",
        metadata={
            "label": "Cache URL",
            "description": "URL to a file with the Wu Palmer cache.",
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


@WU_PALMER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @WU_PALMER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @WU_PALMER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Time tanh endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Wu Palmer similarities",
            description=WuPalmer.instance.description,
            name=WuPalmer.instance.name,
            version=WuPalmer.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{WU_PALMER_BLP.name}.CalcSimilarityView"),
                ui_href=url_for(f"{WU_PALMER_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entities",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesUrl",
                    ),
                    InputDataMetadata(
                        data_type="wu-palmer-cache",
                        content_type=["application/zip"],
                        required=True,
                        parameter="entitiesMetadataUrl",
                    ),
                    InputDataMetadata(
                        data_type="attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                        parameter="wuPalmerCacheUrl",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="element-similarities",
                        content_type=["application/zip"],
                        required=True,
                    )
                ],
            ),
            tags=WuPalmer.instance.tags,
        )


@WU_PALMER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Wu Palmer plugin."""

    @WU_PALMER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Wu Palmer plugin."
    )
    @WU_PALMER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WU_PALMER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @WU_PALMER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Wu Palmer plugin."
    )
    @WU_PALMER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WU_PALMER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=WuPalmer.instance.name,
                version=WuPalmer.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{WU_PALMER_BLP.name}.CalcSimilarityView"),
            )
        )


@WU_PALMER_BLP.route("/process/")
class CalcSimilarityView(MethodView):
    """Start a long running processing task."""

    @WU_PALMER_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @WU_PALMER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @WU_PALMER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name, parameters=dumps(arguments)
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


class WuPalmer(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Compares elements and returns similarity values."
    tags = ["similarity-calculation"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return WU_PALMER_BLP

    def get_requirements(self) -> str:
        return ""


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{WuPalmer.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new Wu Palmer calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    entities_url: Optional[str] = loads(task_data.parameters or "{}").get(
        "entities_url", None
    )
    TASK_LOGGER.info(f"Loaded input parameters from db: entities_url='{entities_url}'")
    entities_metadata_url: Optional[str] = loads(task_data.parameters or "{}").get(
        "entities_metadata_url", None
    )
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entities_metadata_url='{entities_metadata_url}'"
    )
    wu_palmer_cache_url: Optional[str] = loads(task_data.parameters or "{}").get(
        "wu_palmer_cache_url", None
    )
    TASK_LOGGER.info(
        f"Loaded input parameters from db: wu_palmer_cache_url='{wu_palmer_cache_url}'"
    )
    attributes: Optional[str] = loads(task_data.parameters or "{}").get(
        "attributes", None
    )
    TASK_LOGGER.info(f"Loaded input parameters from db: attributes='{attributes}'")
    attributes: List[str] = attributes.splitlines()

    # load data from file

    with open_url(entities_url) as entities_data:
        entities = list(load_entities(entities_data, "application/json"))

    with open_url(entities_metadata_url) as entities_metadata_file:
        entities_metadata_list = list(
            load_entities(entities_metadata_file, "application/json")
        )
        entities_metadata = {element["ID"]: element for element in entities_metadata_list}

    cached_similarities = {}

    for file, file_name in get_files_from_zip_url(wu_palmer_cache_url):
        cached_similarities[file_name[:-5]] = {
            element["ID"]: element for element in json.load(file)
        }

    # calculate similarity values for all possible value pairs

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attribute in attributes:
        similarities = {}

        for i in range(len(entities)):
            for j in range(i, len(entities)):
                ent1 = entities[i]
                ent2 = entities[j]

                if attribute in ent1 and attribute in ent2:
                    values1 = ent1[attribute]
                    values2 = ent2[attribute]

                    # extract taxonomy name from refTarget
                    taxonomy: str = entities_metadata[attribute]["refTarget"].split(":")[
                        1
                    ][:-5]
                    tax_sims = cached_similarities[taxonomy]

                    if not isinstance(values1, list):
                        values1 = [values1]

                    if not isinstance(values2, list):
                        values2 = [values2]

                    for val1 in values1:
                        for val2 in values2:
                            if val1 is None or val2 is None:
                                sim = None
                            else:
                                if str(val1) + "__" + str(val2) in tax_sims:
                                    sim = tax_sims[str(val1) + "__" + str(val2)][
                                        "similarity"
                                    ]
                                elif str(val2) + "__" + str(val1) in tax_sims:
                                    sim = tax_sims[str(val2) + "__" + str(val1)][
                                        "similarity"
                                    ]
                                else:
                                    raise ValueError(
                                        "No similarity value cached for the values "
                                        + str(val1)
                                        + " and "
                                        + str(val2)
                                    )

                            if (val1, val2) not in similarities and (
                                val2,
                                val1,
                            ) not in similarities:
                                similarities[(val1, val2)] = {
                                    "ID": str(val1) + "_" + str(val2),
                                    "href": "",
                                    "value_1": val1,
                                    "value_2": val2,
                                    "similarity": sim,
                                }

        with StringIO() as file:
            save_entities(similarities.values(), file, "application/json")
            file.seek(0)
            zip_file.writestr(attribute + ".json", file.read())

    zip_file.close()

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        "wu_palmer.zip",
        "element-similarities",
        "application/zip",
    )

    return "Result stored in file"
