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

import mimetypes
import random
from enum import Enum
from http import HTTPStatus
from json import JSONEncoder, dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, Generator, List, Mapping, Optional, Set, Union

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from marshmallow.validate import Range

from qhana_plugin_runner.api.extra_fields import CSVList, EnumField
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    InputDataMetadata,
)
from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "entity-filter"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

INFINITY = -1

ENTITY_FILTER_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Entity filter API.",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class AttributeFilterType(Enum):
    ALLOWLIST = "Allowlist"
    BLOCKLIST = "Blocklist"


class RowSamplingType(Enum):
    RANDOM = "Randomly"
    FIRST_N = "First n (Number of Rows)"
    ALL_ROWS = "All Rows"


class EnumEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return JSONEncoder.default(self, obj)


class EntityFilterParametersSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={"label": "Entities URL"},
    )

    attributes = CSVList(
        required=False,
        allow_none=True,
        element_type=ma.fields.String,
        metadata={
            "label": "Attributes",
            "description": "List of attributes in allowlist/blocklist.",
            "input_type": "textarea",
        },
    )

    attribute_filter_strategy = EnumField(
        AttributeFilterType,
        required=True,
        metadata={
            "label": "Attribute Filter Setting",
            "description": "Specify attribute list as allowlist or blocklist.",
            "input_type": "select",
        },
    )

    n_rows = ma.fields.Integer(
        required=False,
        allow_none=True,
        validate=Range(
            0, min_inclusive=False, error="Number of Rows must be a positive number."
        ),
        metadata={
            "label": "Number of Rows",
            "description": "Integer of number of rows that should be kept.",
            "input_type": "textfield",
        },
    )

    row_sampling = EnumField(
        RowSamplingType,
        required=True,
        metadata={
            "label": "Row Sampling",
            "description": "Specify if rows are chosen randomly or first n in case that Number \
                of Rows is set and ID list smaller than Number of Rows. If All Rows is chosen, Number of Rows is ignored.",
            "input_type": "select",
        },
    )

    id_list = CSVList(
        required=False,
        allow_none=True,
        element_type=ma.fields.String,
        metadata={
            "label": "ID List",
            "description": "Comma separated list of ID's that should be kept. If number is \
                smaller than Number of Rows, remaining rows are chosen according to Row Choice.",
            "input_type": "textarea",
        },
    )


@ENTITY_FILTER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @ENTITY_FILTER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @ENTITY_FILTER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Entity filter endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Entity loader/filter",
            description=EntityFilter.instance.description,
            name=EntityFilter.instance.name,
            version=EntityFilter.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{ENTITY_FILTER_BLP.name}.ProcessView"),
                ui_href=url_for(f"{ENTITY_FILTER_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/list",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="inputFileUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/list",
                        content_type=["application/json", "text/csv"],
                        required=True,
                    )
                ],
            ),
            tags=EntityFilter.instance.tags,
        )


@ENTITY_FILTER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the entity filter plugin."""

    example_inputs = {
        "inputFileUrl": "file:///<path_to_file>/entities.json",
        "nRows": 5,
        "attributes": "ID",
    }

    @ENTITY_FILTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the entity filter plugin."
    )
    @ENTITY_FILTER_BLP.arguments(
        EntityFilterParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @ENTITY_FILTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @ENTITY_FILTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the entity filter plugin."
    )
    @ENTITY_FILTER_BLP.arguments(
        EntityFilterParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @ENTITY_FILTER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = EntityFilterParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=EntityFilter.instance.name,
                version=EntityFilter.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{ENTITY_FILTER_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{ENTITY_FILTER_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@ENTITY_FILTER_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @ENTITY_FILTER_BLP.arguments(
        EntityFilterParametersSchema(unknown=EXCLUDE), location="form"
    )
    @ENTITY_FILTER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @ENTITY_FILTER_BLP.require_jwt("jwt", optional=True)
    def post(self, input_params: EntityFilterParametersSchema):
        """Start the entity filter task."""
        db_task = ProcessingTask(
            task_name=entity_filter_task.name,
            parameters=dumps(input_params, cls=EnumEncoder),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = entity_filter_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class EntityFilter(QHAnaPluginBase):

    name = _plugin_name
    version = __version__
    description = (
        "Loads and filters entities from a file that contains a list of entities."
    )
    tags = ["data-loading"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return ENTITY_FILTER_BLP

    def get_requirements(self) -> str:
        return ""


TASK_LOGGER = get_task_logger(__name__)


def filter_rows(
    input_entities: Generator[Dict[str, Any], None, None],
    id_set: Set[str],
    n_sampled_rows: int,
    row_sampling: str,
) -> List[Dict[str, Any]]:
    """Filters rows of ``input_entities``.

    Iterates over entities in ``input_entities``.
    If "ID" of entity is in ``id_set``, the entity is added to output list.
    If not and ``n_sampled_rows > 0``, row sampling is applied according to the strategy specified in ``row_sampling``.
    Random row sampling is done as in `Uniformly sampling from N elements <https://math.stackexchange.com/questions/846036/can-i-uniformly-sample-from-n-distinct-elements-where-n-is-unknown-but-fini>`_.

    Args:
        input_entities (Generator[Dict[str, Any], None, None]): input entities to be filtered
        id_set (Set[str]): list of entity ID's
        n_sampled_rows (int): number of rows that are to be sampled randomly
        row_sampling (str): strategy for sampling (value of :class:`RowSamplingType`)

    Raises:
        ValueError: if invalid value for ``row_sampling``
        ValueError: if some ID's in ``id_set`` cannot be found

    Returns:
        List[Dict[str, Any]]: filtered entities
    """
    # list of output entities with ID in id_set
    output_entities_id_list: List[Dict[str, Any]] = []
    # list of sampled output entities,
    output_entities_random_rows: List[Dict[str, Any]] = []
    # counts number of sampled entities
    sampling_counter = 0
    for entity in input_entities:
        if entity["ID"] in id_set:
            # find entities in id_set if id_set not empty
            output_entities_id_list.append(entity)  # TODO
            id_set.remove(entity["ID"])

        elif n_sampled_rows > 0:
            # sample rows to fill up remaining rows according to row sampling
            if row_sampling == RowSamplingType.RANDOM.value:
                if sampling_counter < n_sampled_rows:
                    # add first n
                    output_entities_random_rows.append(entity)
                    sampling_counter += 1
                else:
                    # add with prob n/(n+k+1) at random index, k is counter
                    if random.random() < n_sampled_rows / (sampling_counter + 1):
                        index = random.randrange(n_sampled_rows)
                        output_entities_random_rows[index] = entity
                    sampling_counter += 1
            elif row_sampling == RowSamplingType.FIRST_N.value:
                if sampling_counter < n_sampled_rows:
                    output_entities_random_rows.append(entity)
                    sampling_counter += 1
            else:
                msg = "Invalid argument for Row Sampling!"
                TASK_LOGGER.error(msg)
                raise ValueError(msg)

    if id_set:  # not all ID's in file
        msg = f"The following ID's could not be found: {str(id_set)}"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    return output_entities_id_list + output_entities_random_rows


def filter_cols(
    input_entities: Union[List[Dict[str, Any]], Generator[Dict[str, Any], None, None]],
    attribute_filter_strategy: Optional[str],
    attributes: Set[str],
) -> Generator[Dict[str, Any], None, None]:
    """Filters colums of ``input_entities``.

    Iterates over all entities and yields output entities filtered as specified in ``attribute_filter_strategy`` and ``attributes``.

    Args:
        input_entities (Union[List[Dict[str, Any]], Generator[Dict[str, Any], None, None]]): input entities to be filtered
        attribute_filter_strategy (Optional[str]): filter strategy as defined in :class:`AttributeFilterType`
        attributes (Set[str]): set of attributes

    Raises:
        ValueError: if attribute attribute filter strategy invalid

    Yields:
        Generator[Dict[str, Any], None, None]: filtered entities
    """
    # make sure that "ID" is not deleted
    if attribute_filter_strategy == AttributeFilterType.ALLOWLIST.value:
        attributes.add("ID")
    elif attribute_filter_strategy == AttributeFilterType.BLOCKLIST.value:
        if "ID" in attributes:
            attributes.remove("ID")
    else:
        msg = "Invalid argument for Attribute Filter Strategy!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    # remove columns that are not in allowlist
    for entity in input_entities:
        if (
            attribute_filter_strategy == AttributeFilterType.ALLOWLIST.value
            and attributes
        ):
            for attr in tuple(entity.keys()):
                if attr not in attributes:
                    del entity[attr]
        else:  # Blocklist
            if attributes:  # nothing to do if empty
                for attr in tuple(entity.keys()):
                    if attr in attributes:
                        del entity[attr]
        yield entity


@CELERY.task(name=f"{EntityFilter.instance.identifier}.entity_filter_task", bind=True)
def entity_filter_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new entity filter task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    input_file_url: Optional[str] = params.get("input_file_url", None)
    # list of attributes
    attributes: Set[str] = set(params.get("attributes", []))
    # choice for attributes (can be either allowlist or blocklist)
    attribute_filter_strategy: Optional[str] = params.get(
        "attribute_filter_strategy", None
    )
    # number of requested output rows
    n_rows: Optional[int] = params.get("n_rows", None)
    # type of sampling (random or first n)
    row_sampling: Optional[str] = params.get("row_sampling", None)
    # set of id's
    id_set: Set[str] = set(params.get("id_list", []))

    TASK_LOGGER.info(
        f"Loaded input parameters from db: input_file_url='{input_file_url}', \
        attributes='{attributes}', attribute_filter_strategy='{attribute_filter_strategy}', n_rows='{n_rows}', \
        row_sampling='{row_sampling}', id_list='{id_set}'"
    )

    ## Check parameter validity ##
    if input_file_url is None or not input_file_url:
        msg = "No input file URL provided!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    # number of rows to be sampled
    n_sampled_rows: int = 0

    if (n_rows is None and not id_set) or row_sampling == RowSamplingType.ALL_ROWS.value:
        # only filter columns
        n_rows = INFINITY
    if not id_set:
        n_sampled_rows = n_rows
    else:
        if n_rows is None:
            n_rows = len(id_set)
        n_sampled_rows = max(0, n_rows - len(id_set))
        if len(id_set) > n_rows:
            msg = "Length of ID list greater than number of rows!"
            TASK_LOGGER.error(msg)
            raise ValueError(msg)

    if row_sampling is None and n_sampled_rows > 0:
        msg = "Row sampling not specified!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    if attribute_filter_strategy is None:
        msg = "Attribute filter strategy not specified!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    ## Filtering ##
    with open_url(input_file_url, stream=True) as url_data:
        r_filtered_entities: Union[
            List[Dict[str, Any]], Generator[Dict[str, Any], None, None]
        ] = []
        try:
            mimetype = url_data.headers["Content-Type"]
        except KeyError:
            mimetype = mimetypes.MimeTypes().guess_type(url=input_file_url)[0]
        input_entities = ensure_dict(load_entities(file_=url_data, mimetype=mimetype))

        # Filter rows
        if n_rows != INFINITY:
            r_filtered_entities = filter_rows(
                input_entities=input_entities,
                id_set=id_set,
                n_sampled_rows=n_sampled_rows,
                row_sampling=row_sampling,
            )

            if len(r_filtered_entities) != n_rows:
                msg = "Number of rows requested is greater than number of rows in input file!"
                TASK_LOGGER.error(msg)
                raise ValueError(msg)
        else:
            r_filtered_entities = input_entities

        # Filter cols
        output_entities = filter_cols(
            input_entities=r_filtered_entities,
            attribute_filter_strategy=attribute_filter_strategy,
            attributes=attributes,
        )

        # Write to output file
        with SpooledTemporaryFile(mode="w") as output:
            save_entities(entities=output_entities, file_=output, mimetype=mimetype)

            if mimetype == "application/json":
                file_type = ".json"
            else:
                file_type = ".csv"
            STORE.persist_task_result(
                db_id,
                output,
                "filtered_entities" + file_type,
                "entity_filter_output",
                mimetype,
            )
    return "Filter successful."  # TODO
