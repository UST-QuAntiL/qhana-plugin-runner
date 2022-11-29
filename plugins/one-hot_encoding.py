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

from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, List, Set, Tuple

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
from itertools import count, chain
import numpy as np


""" 
This Plugin can be further improved!
The below comments were taken from a PR review

Plugin currently works as follows:
    Instead of doing a tree traversal to flatten the tree, the plugin simply uses the entities entry in each taxonomy as 
    the flatten version. The plugin also ignores the "" node, i.e. root, since it provides no additional information.

    Lets assume we have the following taxonomy a[b,c,d[e]] and the entities attribute lists them like this [a, b, c, d, e].
    If we now have an entity that has e as an attribute, then we return the vector [1, 0, 0, 1, 1] for this entity. 
    In other words, we also set the ancestors of e to 1 (the ancestors being a,d).

    The idea of why we also set the ancestors to one becomes clear, when looking at colors. Lets assume the following 
    taxonomy color[red, green, blue[light_blue, dark_blue]. In this case the one-hot encodings of light_blue and dark_blue 
    should be closer to each other, than they are to the one-hot encoding of red. Therefore, we should also set each 
    dimension of our ancestors to 1.
    That is also why the ancestors for each attribute, that we use later on, gets precomputed.

Improvements:
    So you essentially need to store the set of indices to set to one for each vector. 
    (preferably as a tuple but the datastructure used here will not be big anyway).
    Using the already flat entities list in the the graph as tree traversal is OK (but make sure you allow for not 
    only entity ids to be in the entities list [see the refactored wu palmer plugin for how to read in the graph]). 
    The only requirement is that the order is stable (which it should be). The ancestors can be built from the relations 
    of the graph (same code as in the wu-palmer plugin).
    We may want to add an option to which depth we want to flag ancestors (or if we want them at all) to the plugin 
    (if that makes sense for one hot encoding). [if we do this should we count from root or from the node?]


    About the depth idea:
    If I set the depth to n, then we ignore all nodes with an depth < n, right?
    So in our previous example r[a[b,c,d[e]]] with a root r added, if we set the depth to be 1 and we don't count the 
    root r, we get the following one-hot encoding for e [0, 0, 0, 1, 1], i.e. we ignore a. We could go even further and 
    just use [0, 0, 1, 1], completly removing a's dimension. (we always ignore the root).
    Since we always ignore the root, not counting the root makes more sense, but I will add this information to the 
    plugin's description anyways.

    If adding this depth parameter is useful or not, is for the user to determine. Currently I can't think of an example.
"""


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
            "description": "Attributes for which the one-hot encoding shall be computed.",
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
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{ONEHOT_BLP.name}.CalcView"),
                ui_href=url_for(f"{ONEHOT_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entities",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesUrl",
                    ),
                    InputDataMetadata(
                        data_type="taxonomy",
                        content_type=["application/zip"],
                        required=True,
                        parameter="taxonomiesZipUrl",
                    ),
                    InputDataMetadata(
                        data_type="attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesMetadataUrl",
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


TASK_LOGGER = get_task_logger(__name__)


def get_attribute_ref_target(entities_attribute_metadata_url: str, attributes: List[str]):
    result = dict()
    entities_attribute_metadata = open_url(entities_attribute_metadata_url).json()
    for attribute in attributes:
        for metadata in entities_attribute_metadata:
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


def get_entity_dict(id, point):
    ent = {"ID": id, "href": ""}
    for d in range(len(point)):
        ent[f"dim{d}"] = point[d]
    return ent


def taxonomy_node_to_parent(taxonomy):
    """
    Format taxonomy to a dictionary. Given an attribute of the taxonomy, it returns the parent attribute.
    """
    parent_dict = dict()
    for node in taxonomy["relations"]:
        parent_dict[node["target"]] = node["source"]
    return parent_dict


def get_ancestor_nodes(parent_node_dict, attribute, ancestor_nodes_dict) -> Set:
    if attribute == "":
        return set()
    if attribute in ancestor_nodes_dict:
        return ancestor_nodes_dict[attribute]
    else:
        parent = parent_node_dict[attribute]
        result = get_ancestor_nodes(parent_node_dict, parent, ancestor_nodes_dict).copy()
        if parent != "":
            result.add(parent)
        ancestor_nodes_dict[attribute] = result
        return result


def compute_ancestors_and_index_dict(entities, attributes, attribute_ref_targets, taxonomies) -> Tuple[List, List, int]:
    """
    Each entity owns certain attributes in a given taxonomy. This method computes the ancestors for each of the
    attributes in every given taxonomy.
    It also assigns each attribute a unique index (across taxonomies) and it computes the total amount of attributes.
    """
    taxonomies_ancestors_list = []
    attr_to_idx_dict_list = []
    dim = 0
    for attribute in attributes:
        taxonomy = taxonomies[attribute_ref_targets[attribute]]
        parent_node_dict = taxonomy_node_to_parent(taxonomy)

        tax_entities = taxonomy["entities"]
        if tax_entities[0] == "":
            tax_entities = tax_entities[1:]

        attr_to_idx_dict_list.append(dict(zip(tax_entities, count(start=dim))))
        dim += len(tax_entities)

        # A dictionary. Given a node, the dict returns all the ancestry nodes
        ancestor_nodes_dict = dict()
        for entity in entities:
            values = entity[attribute]

            sub_attributes = set()
            if isinstance(values, list):
                sub_attributes = set(values)
            else:
                sub_attributes.add(values)

            for sub_attribute in sub_attributes:
                get_ancestor_nodes(parent_node_dict, sub_attribute, ancestor_nodes_dict)

        taxonomies_ancestors_list.append(ancestor_nodes_dict)

    return taxonomies_ancestors_list, attr_to_idx_dict_list, dim


def prepare_stream_output(entities, attributes, taxonomies_ancestors_list, attr_to_idx_dict_list, dim):
    """
    Transforms an entity into it's one-hot encoding and yields it.
    """
    for entity in entities:
        id = entity["ID"]
        one_hot_encodings = np.zeros((dim, ))
        for attribute, attr_to_idx_dict, taxonomies_ancestors in zip(attributes, attr_to_idx_dict_list, taxonomies_ancestors_list):
            values = entity[attribute]

            sub_attributes = set()
            if isinstance(values, list):
                sub_attributes = set(values)
            else:
                sub_attributes.add(values)

            for sub_attribute in sub_attributes:
                for ancestor in taxonomies_ancestors[sub_attribute]:
                    one_hot_encodings[attr_to_idx_dict[ancestor]] = 1

                one_hot_encodings[attr_to_idx_dict[sub_attribute]] = 1

        yield get_entity_dict(id, one_hot_encodings)


@CELERY.task(name=f"{OneHot.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new one-hot encoding calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entities_url = input_params.entities_url
    entities_attribute_metadata_url = input_params.entities_metadata_url
    taxonomies_zip_url = input_params.taxonomies_zip_url
    attributes = input_params.attributes

    TASK_LOGGER.info(
        f"Loaded input parameters from db: entities_url='{entities_url}', entities_metadata_url='{entities_attribute_metadata_url}', taxonomies_zip_url='{taxonomies_zip_url}', entities_metadata_url='{attributes}'"
    )

    # load data from file
    attributes = attributes.splitlines()
    # ref target is the name of the file containing the taxonomy
    attribute_ref_targets = get_attribute_ref_target(entities_attribute_metadata_url, attributes)
    # load taxonomies
    taxonomies = get_taxonomies_by_ref_target(attribute_ref_targets, taxonomies_zip_url)

    entities = open_url(entities_url).json()
    taxonomies_ancestors_list, attr_to_idx_dict_list, dim = compute_ancestors_and_index_dict(entities, attributes, attribute_ref_targets, taxonomies)
    entity_points = prepare_stream_output(entities, attributes, taxonomies_ancestors_list, attr_to_idx_dict_list, dim)
    csv_attributes = ["ID", "href"] + [f"dim{d}" for d in range(dim)]

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
