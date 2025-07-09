import json
import time
from json import loads
from tempfile import SpooledTemporaryFile
from typing import Optional

import numpy as np
from celery.utils.log import get_task_logger
from common.algorithms import compute_schmidt_rank
from common.plugin_utils.task_util import generate_numpy_vectors
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

from . import Plugin

TASK_LOGGER = get_task_logger(__name__)

_taskname_ = "schmidt_rank"


@CELERY.task(
    name=f"{Plugin.instance.identifier}.{_taskname_}_task",
    bind=True,
)
def task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting '{_taskname_}' task with DB ID='{db_id}'.")

    task_data = ProcessingTask.get_by_id(id_=db_id)
    if not task_data:
        msg = f"No task data found for ID {db_id}."
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = loads(task_data.parameters or "{}")
    TASK_LOGGER.info(f"Parameters: {params}")

    ##
    # Get inputs
    tolerance = params.get("tolerance", None)
    vector = params.get("vector", None)
    qasm_input_list = params.get("qasmInputList", None)
    dimHX = params.get("dimHX")
    dimHR = params.get("dimHR")
    if vector:
        np_vectors = generate_numpy_vectors(qasm_input_list, [vector])
    else:
        np_vectors = generate_numpy_vectors(qasm_input_list, None)

    TASK_LOGGER.info(
        f"Received vectors: {np_vectors} dimHx: {dimHX}, dimHr: {dimHR}, tolerance,{tolerance}"
    )

    if len(np_vectors) != 1:
        raise Exception(
            f"Invalid input: The calculation process only takes one state/vector, but {len(np_vectors)} were provided."
        )
    np_vector = np_vectors[0]

    # Check inputs
    dim = len(np_vector)

    if dim != (2 ** (dimHR + dimHX)):
        raise Exception(
            f"Invalid input vector dimensions for {_taskname_} calculation. "
            f"The input vector must have a dimension of 2^(dimHR + dimHX) = {2 ** (dimHR + dimHX)}, "
            f"but the provided vectors have a dimension of {dim}. "
            f"Ensure that dimHR ({dimHR}) and dimHX ({dimHX}) are correctly defined and that the input vector match "
            f"the expected dimension for a bipartite quantum system."
        )

    ##
    # Perform state analysis
    try:
        start_time = time.time()
        ##
        result = compute_schmidt_rank(np_vector, dimHX, dimHR, tolerance)
        ##
        end_time = time.time()

        elapsed_time = end_time - start_time

        output_data = {
            "result": int(result),
            "inputType": "vectors" if vector else "circuit",
            "calculationTime": elapsed_time,
        }

        # Save results
        with SpooledTemporaryFile(mode="w") as json_file:
            json.dump(output_data, json_file)
            json_file.seek(0)
            STORE.persist_task_result(
                db_id,
                json_file,
                "out.json",
                f"custom/{_taskname_}-output",
                "application/json",
            )

        TASK_LOGGER.info(f"{_taskname_} result: {output_data}")
        return json.dumps(output_data)

    except Exception as e:
        TASK_LOGGER.error(f"Error in '{_taskname_}' task: {e}")
        raise
