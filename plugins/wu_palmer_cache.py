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
import os
from http import HTTPStatus
from itertools import combinations
from json import dumps, loads
from tempfile import SpooledTemporaryFile, NamedTemporaryFile
from typing import Mapping, Optional, Dict, Tuple
from zipfile import ZipFile

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
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "wu-palmer-cache"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


WU_PALMER_CACHE_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Wu-Palmer cache plugin API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class CalcSimilarityParametersSchema(FrontendFormBaseSchema):
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


@WU_PALMER_CACHE_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @WU_PALMER_CACHE_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @WU_PALMER_CACHE_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Wu Palmer cache endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Wu-Palmer cache generator",
            description=WuPalmerCache.instance.description,
            name=WuPalmerCache.instance.name,
            version=WuPalmerCache.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{WU_PALMER_CACHE_BLP.name}.CalcSimilarityView"),
                ui_href=url_for(f"{WU_PALMER_CACHE_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="taxonomy",
                        content_type=["application/zip"],
                        required=True,
                        parameter="taxonomiesZipUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="wu-palmer-cache",
                        content_type=["application/zip"],
                        required=True,
                    )
                ],
            ),
            tags=WuPalmerCache.instance.tags,
        )


@WU_PALMER_CACHE_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Wu Palmer cache plugin."""

    @WU_PALMER_CACHE_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Wu Palmer cache plugin."
    )
    @WU_PALMER_CACHE_BLP.arguments(
        CalcSimilarityParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WU_PALMER_CACHE_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @WU_PALMER_CACHE_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Wu Palmer cache plugin."
    )
    @WU_PALMER_CACHE_BLP.arguments(
        CalcSimilarityParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WU_PALMER_CACHE_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = CalcSimilarityParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=WuPalmerCache.instance.name,
                version=WuPalmerCache.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{WU_PALMER_CACHE_BLP.name}.CalcSimilarityView"),
            )
        )


@WU_PALMER_CACHE_BLP.route("/process/")
class CalcSimilarityView(MethodView):
    """Start a long running processing task."""

    @WU_PALMER_CACHE_BLP.arguments(
        CalcSimilarityParametersSchema(unknown=EXCLUDE), location="form"
    )
    @WU_PALMER_CACHE_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @WU_PALMER_CACHE_BLP.require_jwt("jwt", optional=True)
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


class WuPalmerCache(QHAnaPluginBase):

    name = _plugin_name
    version = __version__
    description = "Generates a cache of similarity values based on a taxonomy."
    tags = ["similarity-cache-generation"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return WU_PALMER_CACHE_BLP

    def get_requirements(self) -> str:
        return ""


TASK_LOGGER = get_task_logger(__name__)


def load_taxonomy(taxonomy: Dict) -> Dict[str, Tuple[str, ...]]:
    nodes: Dict[str, Tuple[str, ...]] = {}
    edges: Dict[str, str] = {}
    root_node: Optional[str] = ""

    if taxonomy["type"] != "tree":
        return

    for relation in taxonomy["relations"]:
        edges[relation["target"]] = relation["source"]

    for node in taxonomy["entities"]:
        node_id: str

        if isinstance(node, str):
            node_id = node
        elif isinstance(node, dict):
            node_id = node["ID"]
        elif isinstance(node, tuple):
            node_id = node[0]
        else:
            continue  # TODO: warning

        ancestors = [node_id]
        visited = {node_id}
        current_node_id = node_id

        for _ in range(len(edges)):
            current_node_id = edges.get(current_node_id, None)

            if current_node_id is None:
                break

            if current_node_id in visited:
                return None  # TODO: warning

            ancestors.append(current_node_id)
            visited.add(current_node_id)

        if root_node is not None:
            if root_node == ancestors[-1]:
                root_node = None

            if not root_node:
                root_node = ancestors[-1]

        nodes[node_id] = tuple(ancestors)[::-1]

    return nodes


def wu_palmer_generator(nodes: Dict[str, Tuple[str, ...]]):
    node_list = sorted(nodes.keys())

    for node_a, node_b in combinations(node_list, 2):
        ancestors_a = nodes[node_a]
        ancestors_b = nodes[node_b]

        assert ancestors_a[0] == ancestors_b[0]

        common_ancestor_depth = -1

        for a, b in zip(ancestors_a, ancestors_b):
            if a != b:
                break

            common_ancestor_depth += 1

        denominator = len(ancestors_a) + len(ancestors_b) - 2

        if denominator == 0:
            yield node_a, node_b, 1
        else:
            wu_palmer_similarity = (2 * common_ancestor_depth) / denominator

            yield node_a, node_b, wu_palmer_similarity


@CELERY.task(name=f"{WuPalmerCache.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new Wu Palmer calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    taxonomies_zip_url: Optional[str] = loads(task_data.parameters or "{}").get(
        "taxonomies_zip_url", None
    )
    TASK_LOGGER.info(
        f"Loaded input parameters from db: taxonomies_zip_url='{taxonomies_zip_url}'"
    )

    # load data from file

    taxonomies = {}

    for zipped_file, file_name in get_files_from_zip_url(taxonomies_zip_url, mode="t"):
        taxonomy = json.load(zipped_file)
        taxonomies[file_name[:-5]] = taxonomy

    # calculate similarity values for all taxonomies and all possible value pairs

    TASK_LOGGER.info("Start creating caches")
    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for tax_name in taxonomies.keys():
        tax = load_taxonomy(taxonomies[tax_name])

        if tax is None:
            TASK_LOGGER.warning(f"taxonomy {tax_name} could not be loaded correctly")
            print(f"tax {tax_name} is none")
            continue

        try:
            tax_file = NamedTemporaryFile(mode="wt", delete=False)

            save_entities(
                wu_palmer_generator(tax),
                tax_file,
                mimetype="text/csv",
                attributes=["source", "target", "wu_palmer"],
            )

            tax_file.close()
            zip_file.write(tax_file.name, f"{tax_name}.csv")
        finally:
            os.unlink(tax_file.name)

    zip_file.close()

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        "wu_palmer_cache.zip",
        "wu-palmer-cache",
        "application/zip",
    )

    return "Result stored in file"


def test_wu_palmer_generator():
    nodes1 = {
        "y": ("w", "1", "2", "3.1", "4.1", "y"),
        "x": ("w", "1", "2", "3.2", "4.2", "x"),
    }
    nodes2 = {
        "y": ("w", "1", "2.1", "y"),
        "x": ("w", "1", "2.2", "3.2", "x"),
    }
    nodes3 = {
        "y": ("w", "1.1", "2.1", "y"),
        "x": ("w", "1.2", "x"),
    }

    assert list(wu_palmer_generator(nodes1))[0][2] == 2 / 5
    assert list(wu_palmer_generator(nodes2))[0][2] == 2 / 7
    assert list(wu_palmer_generator(nodes3))[0][2] == 0
