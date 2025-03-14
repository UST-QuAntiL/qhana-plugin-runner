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
from functools import lru_cache
from http import HTTPStatus
from io import StringIO
from json import dumps, loads
from pathlib import PurePath
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, List, Dict, Tuple, Callable
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
    SecurityBlueprint,
    FileUrl,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import (
    tuple_deserializer,
    AttributeMetadata,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
)
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import open_url, retrieve_filename, get_mimetype
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "wu-palmer"
__version__ = "v0.2.1"
_identifier = plugin_identifier(_plugin_name, __version__)


WU_PALMER_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Wu Palmer plugin API.",
)


class InputParametersSchema(FrontendFormBaseSchema):
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/list",
        data_content_types=["text/csv", "application/json"],
        metadata={
            "label": "Entities URL",
            "description": "URL to a file with entities.",
            "input_type": "text",
        },
    )
    entities_metadata_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/attribute-metadata",
        data_content_types="application/json",
        metadata={
            "label": "Entities Attribute Metadata URL",
            "description": "URL to a file with the attribute metadata for the entities.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "post",
        },
    )
    taxonomies_zip_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="graph/taxonomy",
        data_content_types="application/zip",
        metadata={
            "label": "Taxonomies URL",
            "description": "URL to zip file with taxonomies.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "pre",
        },
    )
    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Attributes",
            "description": "List of attributes for which the similarity shall be computed. Separated by newlines.",
            "input_type": "textarea",
        },
    )
    root_is_part_of_hierarchy = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Consider root node as part of the hierarchy",
            "description": "If the root node is part of the hierarchy, then items that are direct descendants of the "
            "root node are considered similar to a certain degree. Otherwise they will be considered as not similar. "
            "e.g. when the root node of a color taxonomy also represents a color, it should be considered as part of "
            "the hierarchy",
            "input_type": "checkbox",
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
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{WU_PALMER_BLP.name}.CalcSimilarityView"),
                ui_href=url_for(f"{WU_PALMER_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/list",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesUrl",
                    ),
                    InputDataMetadata(
                        data_type="entity/attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesMetadataUrl",
                    ),
                    InputDataMetadata(
                        data_type="graph/taxonomy",
                        content_type=["application/zip"],
                        required=True,
                        parameter="taxonomiesZipUrl",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="custom/element-similarities",
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
        return self.render(request.args, errors, False)

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
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=WuPalmer.instance.name,
                version=WuPalmer.instance.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{WU_PALMER_BLP.name}.CalcSimilarityView"),
            )
        )


@WU_PALMER_BLP.route("/process/")
class CalcSimilarityView(MethodView):
    """Start a long running processing task."""

    @WU_PALMER_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @WU_PALMER_BLP.response(HTTPStatus.SEE_OTHER)
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
    tags = ["preprocessing", "similarity-calculation"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return WU_PALMER_BLP

    def get_requirements(self) -> str:
        return "muid~=0.5.3"


TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    import muid

    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def load_taxonomy_as_node_paths(taxonomy: Dict) -> Dict[str, Tuple[str, ...]]:
    nodes: Dict[str, Tuple[str, ...]] = {}
    edges: Dict[str, str] = {}
    root_node: Optional[str] = ""

    if taxonomy["type"] != "tree":
        TASK_LOGGER.warn("taxonomy is not a tree")
        return None

    for relation in taxonomy["relations"]:
        if relation["target"] != relation["source"]:
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
            TASK_LOGGER.warn("entity has an unsupported type")
            continue

        ancestors = [node_id]
        visited = {node_id}
        current_node_id = node_id

        for _ in range(len(edges)):
            current_node_id = edges.get(current_node_id, None)

            if current_node_id is None:
                break

            if current_node_id in visited:
                TASK_LOGGER.warn(
                    f"cycle detected, node {current_node_id} already visited"
                )
                return None

            ancestors.append(current_node_id)
            visited.add(current_node_id)

        if root_node is not None:
            if root_node == ancestors[-1]:
                root_node = None

            if not root_node:
                root_node = ancestors[-1]

        nodes[node_id] = tuple(ancestors)[::-1]

    return nodes


class WuPalmerCache:
    def __init__(self, taxonomies: Dict[str, Dict], root_has_meaning_in_taxonomy: bool):
        self._taxonomies = taxonomies
        self._node_path_cache: Dict[str, Dict[str, Tuple[str, ...]]] = {}

        if root_has_meaning_in_taxonomy:
            self._root_node_depth = 0
        else:
            self._root_node_depth = 1

    @lru_cache
    def calculate_similarity(self, tax_name: str, node_a: str, node_b: str) -> float:
        if tax_name not in self._node_path_cache:
            self._node_path_cache[tax_name] = load_taxonomy_as_node_paths(
                self._taxonomies[tax_name]
            )

        node_paths = self._node_path_cache[tax_name]

        if node_a not in node_paths:
            raise KeyError(f"node {node_a} not in node paths of taxonomy {tax_name}")

        if node_b not in node_paths:
            raise KeyError(f"node {node_b} not in node paths of taxonomy {tax_name}")

        ancestors_a = node_paths[node_a]
        ancestors_b = node_paths[node_b]

        common_ancestor_depth = 0
        common_ancestor_depth -= self._root_node_depth

        for a, b in zip(ancestors_a, ancestors_b):
            if a != b:
                break

            common_ancestor_depth += 1

        denominator = len(ancestors_a) + len(ancestors_b) - (2 * self._root_node_depth)

        if denominator == 0:
            return 1
        else:
            wu_palmer_similarity = (2 * common_ancestor_depth) / denominator

            return wu_palmer_similarity


def add_similarities_for_entities(
    similarities: Dict[Tuple[any, any], Dict],
    entity1: Dict[str, any],
    entity2: Dict[str, any],
    attribute: str,
    entities_metadata: dict[str, AttributeMetadata],
    wu_palmer_cache: WuPalmerCache,
):
    if attribute not in entity1 or attribute not in entity2:
        return

    values1 = entity1[attribute]
    values2 = entity2[attribute]

    # extract taxonomy name from refTarget
    file_name: str = entities_metadata[attribute].ref_target.split(":")[1]
    tax_name: str = PurePath(file_name).stem

    if isinstance(values1, set):
        values1 = list(values1)

    if isinstance(values2, set):
        values2 = list(values2)

    if not isinstance(values1, list):
        values1 = [values1]

    if not isinstance(values2, list):
        values2 = [values2]

    for val1 in values1:
        if not val1:
            continue

        for val2 in values2:
            if not val2:
                continue

            # sorting the values reduces cache misses and is possible because Wu-Palmer is commutative
            sorted_val1, sorted_val2 = sorted((val1, val2))
            sim = wu_palmer_cache.calculate_similarity(tax_name, sorted_val1, sorted_val2)

            similarities[(val1, val2)] = {
                "source": val1,
                "target": val2,
                "similarity": sim,
            }


def load_input_parameters(
    db_id: int,
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str], Optional[bool]]:
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
    taxonomies_zip_url: Optional[str] = loads(task_data.parameters or "{}").get(
        "taxonomies_zip_url", None
    )
    TASK_LOGGER.info(
        f"Loaded input parameters from db: taxonomies_zip_url='{taxonomies_zip_url}'"
    )
    attributes: Optional[str] = loads(task_data.parameters or "{}").get(
        "attributes", None
    )
    TASK_LOGGER.info(f"Loaded input parameters from db: attributes='{attributes}'")
    attributes: List[str] = attributes.splitlines()
    root_has_meaning_in_taxonomy: Optional[bool] = loads(
        task_data.parameters or "{}"
    ).get("root_has_meaning_in_taxonomy", None)
    TASK_LOGGER.info(
        f"Loaded input parameters from db: root_has_meaning_in_taxonomy='{root_has_meaning_in_taxonomy}'"
    )

    return (
        entities_url,
        entities_metadata_url,
        taxonomies_zip_url,
        attributes,
        root_has_meaning_in_taxonomy,
    )


@CELERY.task(name=f"{WuPalmer.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    (
        entities_url,
        entities_metadata_url,
        taxonomies_zip_url,
        attributes,
        root_has_meaning_in_taxonomy,
    ) = load_input_parameters(db_id)
    deserializer: Callable[[tuple[str, ...]], tuple[any, ...]] | None = None

    with open_url(entities_metadata_url) as entities_metadata_file:
        entities_metadata_list = list(
            load_entities(entities_metadata_file, get_mimetype(entities_metadata_file))
        )
        entities_metadata = {
            element["ID"]: AttributeMetadata.from_dict(element)
            for element in entities_metadata_list
        }

    # load data from file
    with open_url(entities_url) as entities_data:
        mimetype = get_mimetype(entities_data)
        entities = []

        for ent in load_entities(entities_data, mimetype):
            if hasattr(ent, "_asdict"):  # is NamedTuple
                if deserializer is None:
                    ent_attributes: tuple[str, ...] = ent._fields
                    ent_tuple = type(ent)
                    deserializer = tuple_deserializer(
                        ent_attributes, entities_metadata, tuple_=ent_tuple._make
                    )

                ent = deserializer(ent)
                entities.append(ent._asdict())
            else:
                entities.append(ent)

    taxonomies = {}

    for zipped_file, file_name in get_files_from_zip_url(taxonomies_zip_url, mode="t"):
        tax_name: Dict = json.load(zipped_file)
        taxonomies[file_name[:-5]] = tax_name

    wu_palmer_cache = WuPalmerCache(taxonomies, root_has_meaning_in_taxonomy)

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attribute in attributes:
        similarities = {}

        for i in range(len(entities)):
            for j in range(i, len(entities)):
                add_similarities_for_entities(
                    similarities,
                    entities[i],
                    entities[j],
                    attribute,
                    entities_metadata,
                    wu_palmer_cache,
                )

        with StringIO() as file:
            save_entities(similarities.values(), file, "application/json")
            file.seek(0)
            zip_file.writestr(attribute + ".json", file.read())

    zip_file.close()

    concat_filenames = retrieve_filename(entities_url)
    concat_filenames += retrieve_filename(entities_metadata_url)
    concat_filenames += retrieve_filename(taxonomies_zip_url)
    filenames_hash = get_readable_hash(concat_filenames)

    info_str = "_with_root" if root_has_meaning_in_taxonomy else "_without_root"
    info_str += f"_{filenames_hash}"

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        f"wu_palmer{info_str}.zip",
        "custom/element-similarities",
        "application/zip",
    )

    return "Result stored in file"
