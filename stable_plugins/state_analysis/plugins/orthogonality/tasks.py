import json
import time
from json import loads
from tempfile import SpooledTemporaryFile
from typing import Optional

import numpy as np
from celery.utils.log import get_task_logger
from common.algorithms import are_vectors_orthogonal
from common.plugin_utils.task_util import generate_numpy_vectors
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

from . import Plugin

TASK_LOGGER = get_task_logger(__name__)

_taskname_ = "orthogonality"


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
    vectors = params.get("vectors", None)
    qasm_input_list = params.get("qasmInputList", None)
    np_vectors = generate_numpy_vectors(qasm_input_list, vectors)

    # Check inputs
    if len(np_vectors) != 2:
        raise Exception(
            f"The {_taskname_} expects 2 vectors, but {len(np_vectors)} were given."
        )

    dim = len(np_vectors[0])
    for np_vector in np_vectors:
        if len(np_vector) != dim:
            raise Exception(
                f"All input vectors for {_taskname_} calculation must have the same dimension. "
                f"The first vector had a dimension of {dim}, but the vector {np_vector} had a dimension of {len(np_vector)}."
            )
    ##
    # Perform state analysis
    try:
        start_time = time.time()
        ##
        result = are_vectors_orthogonal(np_vectors[0], np_vectors[1], tolerance=tolerance)
        ##
        end_time = time.time()

        elapsed_time = end_time - start_time

        output_data = {
            "result": bool(result),
            "inputType": "vectors" if vectors else "circuit",
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
