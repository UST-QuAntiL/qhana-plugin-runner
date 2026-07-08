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
from io import StringIO
from tempfile import SpooledTemporaryFile
from typing import Optional, List
from zipfile import ZipFile
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import retrieve_filename

from . import ElementTransformers
from .schemas import InputParameters, InputParametersSchema, TransformersEnum

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{ElementTransformers.instance.identifier}.calculation_task", bind=True
)
def calculation_task(self, db_id: int) -> str:
    # get parameters
    TASK_LOGGER.info(
        f"Starting new Element Distances calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    similarities_url = input_params.similarities_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: similarities_url='{similarities_url}'"
    )

    attributes: str = input_params.attributes
    TASK_LOGGER.info(f"Loaded input parameters from db: attributes='{attributes}'")
    attributes: List[str] = attributes.splitlines()
    transformer = input_params.transformer
    TASK_LOGGER.info(f"Loaded input parameters from db: transformer='{transformer}'")

    # Load Data From File
    element_similarities = {}

    for file, file_name in get_files_from_zip_url(similarities_url):
        # removes .json from file name to get the name of the attribute
        attr_name = file_name[:-5]
        element_similarities[attr_name] = json.load(file)

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attribute in attributes:
        element_distances = []
        attribute_element_sims = element_similarities[attribute]

        for sim_entity in attribute_element_sims:
            sim = sim_entity.get("similarity")
            dist = None

            # Apply transformation algorithms
            if sim is None:
                dist = None
            elif transformer == TransformersEnum.linear_inverse:
                dist = 1.0 - sim
            elif transformer == TransformersEnum.exponential_inverse:
                dist = math.exp(-sim)
            elif transformer == TransformersEnum.gaussian_inverse:
                dist = math.exp(-sim * sim)
            elif transformer == TransformersEnum.polynomial_inverse:
                alpha = 1.0
                beta = 1.0
                dist = 1.0 / (1.0 + pow(sim / alpha, beta))
            elif transformer == TransformersEnum.square_inverse:
                max_sim = 1.0
                dist = (1.0 / math.sqrt(2.0)) * math.sqrt(2.0 * max_sim - 2 * sim)

            dist_entity = sim_entity.copy()

            # Replace the 'similarity' key with 'distance'
            if "similarity" in dist_entity:
                del dist_entity["similarity"]
            dist_entity["distance"] = dist

            element_distances.append(dist_entity)

        # Save Output
        with StringIO() as file:
            save_entities(element_distances, file, "application/json")
            file.seek(0)
            zip_file.writestr(attribute + ".json", file.read())

    zip_file.close()

    filename = retrieve_filename(similarities_url)
    info_str = f"_transformer_{transformer.name}_from_{filename}"

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        f"transformers_elem_dist{info_str}.zip",
        "custom/element-distances",
        "application/zip",
    )

    return "Result stored in file"
