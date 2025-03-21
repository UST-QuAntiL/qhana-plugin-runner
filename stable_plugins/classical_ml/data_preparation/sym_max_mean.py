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
from typing import Mapping, Optional, List, Dict, Callable
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

_plugin_name = "sym-max-mean"
__version__ = "v0.1.2"
_identifier = plugin_identifier(_plugin_name, __version__)


SYM_MAX_MEAN_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Sym Max Mean plugin API.",
)


class InputParametersSchema(FrontendFormBaseSchema):
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/list",
        data_content_types=["application/json", "text/csv"],
        metadata={
            "label": "Entities URL",
            "description": "URL to a file with entities.",
            "input_type": "text",
        },
    )
    element_similarities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="custom/element-similarities",
        data_content_types="application/zip",
        metadata={
            "label": "Element similarities URL",
            "description": "URL to a zip file with the element similarities for the entities.",
            "input_type": "text",  #
            "related_to": "entities_url",
            "relation": "post",
        },
    )
    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Attributes",
            "description": "Attributes that shall be compared with Sym Max Mean.",
            "input_type": "textarea",
        },
    )


@SYM_MAX_MEAN_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @SYM_MAX_MEAN_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @SYM_MAX_MEAN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Sym Max Mean endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Sym Max Mean attribute comparer",
            description=SymMaxMean.instance.description,
            name=SymMaxMean.instance.name,
            version=SymMaxMean.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{SYM_MAX_MEAN_BLP.name}.CalcSimilarityView"),
                ui_href=url_for(f"{SYM_MAX_MEAN_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/list",
                        content_type=["application/json", "text/csv"],
                        required=True,
                        parameter="entitiesUrl",
                    ),
                    InputDataMetadata(
                        data_type="custom/element-similarities",
                        content_type=["application/zip"],
                        required=True,
                        parameter="elementSimilaritiesUrl",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="custom/attribute-similarities",
                        content_type=["application/zip"],
                        required=True,
                    )
                ],
            ),
            tags=SymMaxMean.instance.tags,
        )


@SYM_MAX_MEAN_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Sym Max Mean plugin."""

    @SYM_MAX_MEAN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Sym Max Mean plugin."
    )
    @SYM_MAX_MEAN_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @SYM_MAX_MEAN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @SYM_MAX_MEAN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Sym Max Mean plugin."
    )
    @SYM_MAX_MEAN_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @SYM_MAX_MEAN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=SymMaxMean.instance.name,
                version=SymMaxMean.instance.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{SYM_MAX_MEAN_BLP.name}.CalcSimilarityView"),
            )
        )


@SYM_MAX_MEAN_BLP.route("/process/")
class CalcSimilarityView(MethodView):
    """Start a long running processing task."""

    @SYM_MAX_MEAN_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @SYM_MAX_MEAN_BLP.response(HTTPStatus.SEE_OTHER)
    @SYM_MAX_MEAN_BLP.require_jwt("jwt", optional=True)
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


class SymMaxMean(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Compares attributes and returns similarity values."
    tags = ["preprocessing", "similarity-calculation"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return SYM_MAX_MEAN_BLP

    def get_requirements(self) -> str:
        return "muid~=0.5.3"


TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    import muid

    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def _get_sim(elem_sims: Dict, val1, val2) -> float:
    if (val1, val2) in elem_sims:
        return elem_sims[(val1, val2)]["similarity"]
    elif (val2, val1) in elem_sims:
        return elem_sims[(val2, val1)]["similarity"]
    else:
        return 0.0  # handles missing elements in lists


@CELERY.task(name=f"{SymMaxMean.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new Sym Max Mean calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    entities_url: Optional[str] = loads(task_data.parameters or "{}").get(
        "entities_url", None
    )
    TASK_LOGGER.info(f"Loaded input parameters from db: entities_url='{entities_url}'")
    element_similarities_url: Optional[str] = loads(task_data.parameters or "{}").get(
        "element_similarities_url", None
    )
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entities_similarities_url='{element_similarities_url}'"
    )
    attributes: Optional[str] = loads(task_data.parameters or "{}").get(
        "attributes", None
    )
    TASK_LOGGER.info(f"Loaded input parameters from db: attributes='{attributes}'")
    attributes: List[str] = attributes.splitlines()

    # load data from file

    with open_url(entities_url) as entities_data:
        mimetype = get_mimetype(entities_data)
        entities = []
        deserializer: Callable[[tuple[str, ...]], tuple[any, ...]] | None = None
        attribute_metadata: dict[str, AttributeMetadata] | None = None

        if "X-Attribute-Metadata" in entities_data.headers:
            attribute_metadata_url = entities_data.headers["X-Attribute-Metadata"]
            attribute_metadata_list = open_url(attribute_metadata_url).json()
            attribute_metadata = {}

            for attr_meta in attribute_metadata_list:
                attribute_metadata[attr_meta["ID"]] = AttributeMetadata.from_dict(
                    attr_meta
                )

        for ent in load_entities(entities_data, mimetype):
            if hasattr(ent, "_asdict"):  # is NamedTuple
                ent_attributes: tuple[str, ...] = ent._fields
                ent_tuple = type(ent)

                if deserializer is None and attribute_metadata is not None:
                    deserializer = tuple_deserializer(
                        ent_attributes, attribute_metadata, tuple_=ent_tuple._make
                    )

                if deserializer:
                    ent = deserializer(ent)

                entities.append(ent._asdict())
            else:
                entities.append(ent)

    element_similarities = {}

    for file, file_name in get_files_from_zip_url(element_similarities_url):
        # removes .json from file name to get the name of the attribute
        attr_name = file_name[:-5]

        element_similarities[attr_name] = {
            (element["source"], element["target"]): element for element in json.load(file)
        }

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attribute in attributes:
        elem_sims = element_similarities[attribute]
        attribute_similarities = []

        for i in range(len(entities)):
            for j in range(i, len(entities)):
                ent1 = entities[i]
                ent2 = entities[j]

                ent_attr1 = ent1[attribute]
                ent_attr2 = ent2[attribute]

                if ent_attr1 is None or ent_attr2 is None:
                    sym_max_mean = None  # TODO: add handling of missing values
                elif len(ent_attr1) == 0 and len(ent_attr2) == 0:
                    sym_max_mean = None
                elif len(ent_attr1) == 0 or len(ent_attr2) == 0:
                    sym_max_mean = 0
                else:
                    if isinstance(ent_attr1, set):
                        ent_attr1 = list(ent_attr1)

                    if isinstance(ent_attr2, set):
                        ent_attr2 = list(ent_attr2)

                    if not isinstance(ent_attr1, list):
                        ent_attr1 = [ent_attr1]

                    if not isinstance(ent_attr2, list):
                        ent_attr2 = [ent_attr2]

                    # calculate Sym Max Mean

                    sum1 = 0.0
                    sum2 = 0.0

                    for a in ent_attr1:
                        # get maximum similarity
                        max_sim = 0.0

                        for b in ent_attr2:
                            sim = _get_sim(elem_sims, a, b)

                            if sim > max_sim:
                                max_sim = sim

                        sum1 += max_sim

                    # calculate the average of the maximum similarities
                    avg1 = sum1 / len(ent_attr1)

                    for b in ent_attr2:
                        max_sim = 0.0

                        for a in ent_attr1:
                            sim = _get_sim(elem_sims, b, a)

                            if sim > max_sim:
                                max_sim = sim

                        sum2 += max_sim

                    # calculate the average of the maximum similarities
                    avg2 = sum2 / len(ent_attr2)

                    sym_max_mean = (avg1 + avg2) / 2.0

                attribute_similarities.append(
                    {
                        "ID": ent1["ID"] + "__" + ent2["ID"] + "__" + attribute,
                        "entity_1_ID": ent1["ID"],
                        "entity_2_ID": ent2["ID"],
                        "href": "",
                        "similarity": sym_max_mean,
                    }
                )

        with StringIO() as file:
            save_entities(attribute_similarities, file, "application/json")
            file.seek(0)
            zip_file.writestr(attribute + ".json", file.read())

    zip_file.close()

    concat_filenames = retrieve_filename(entities_url)
    concat_filenames += retrieve_filename(element_similarities_url)
    filenames_hash = get_readable_hash(concat_filenames)
    info_str = f"_{filenames_hash}"

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        f"sym_max_mean{info_str}.zip",
        "custom/attribute-similarities",
        "application/zip",
    )

    return "Result stored in file"
