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
import math
from http import HTTPStatus
from io import BytesIO, StringIO
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional
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
        return "networkx~=2.5.1"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{WuPalmerCache.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    import networkx as nx

    def compare_inner(first: str, second: str, graph: nx.DiGraph) -> float:
        """
        Applies wu palmer similarity measure on two taxonomie elements
        """

        # Get undirected graph
        ud_graph = graph.to_undirected()

        # Get lowest reachable node from both
        lowest_common_ancestor = (
            nx.algorithms.lowest_common_ancestors.lowest_common_ancestor(
                graph, first, second
            )
        )

        # Get root of graph
        root = [n for n, d in graph.in_degree() if d == 0][0]

        # Count edges - weight is 1 per default
        d1 = nx.algorithms.shortest_paths.generic.shortest_path_length(
            ud_graph, first, lowest_common_ancestor
        )
        d2 = nx.algorithms.shortest_paths.generic.shortest_path_length(
            ud_graph, second, lowest_common_ancestor
        )
        d3 = nx.algorithms.shortest_paths.generic.shortest_path_length(
            ud_graph, lowest_common_ancestor, root
        )

        # if first and second, both is the root
        if d1 + d2 + 2 * d3 == 0.0:
            return 0.0

        return 2 * d3 / (d1 + d2 + 2 * d3)

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
        taxonomy = taxonomies[tax_name]
        TASK_LOGGER.info("Start caching " + tax_name)

        # create graph from taxonomy
        graph = nx.DiGraph()

        for entity in taxonomy["entities"]:
            graph.add_node(entity)

        for relation in taxonomy["relations"]:
            if not relation["source"] == relation["target"]:
                graph.add_edge(relation["source"], relation["target"])

        amount = int(math.pow(len(taxonomy["entities"]), 2) / 2)
        index = 1
        every_n_steps = 100
        cache = []

        nodes = list(graph.nodes())

        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                first = nodes[i]
                second = nodes[j]

                cache.append(
                    {
                        "ID": first + "__" + second,
                        "first_node": first,
                        "second_node": second,
                        "similarity": compare_inner(first, second, graph),
                    }
                )
                index += 1

                if index % every_n_steps == 0:
                    TASK_LOGGER.info(str(index) + " from " + str(amount))

        with StringIO() as file:
            save_entities(cache, file, "application/json")
            file.seek(0)
            zip_file.writestr(tax_name + ".json", file.read())

    zip_file.close()

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        "wu_palmer_cache.zip",
        "wu-palmer-cache",
        "application/zip",
    )

    return "Result stored in file"
