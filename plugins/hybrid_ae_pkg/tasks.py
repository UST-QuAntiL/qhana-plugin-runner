from json import loads
from tempfile import SpooledTemporaryFile
from typing import Dict

from celery.utils.log import get_task_logger
from sqlalchemy import select

from . import HybridAutoencoderPlugin
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{HybridAutoencoderPlugin.instance.identifier}.pennylane_hybrid_autoencoder_task",
    bind=True,
)
def hybrid_autoencoder_pennylane_task(self, db_id: int) -> str:
    import numpy as np
    from .backend import simple_api

    TASK_LOGGER.info(
        f"Starting new hybrid autoencoder pennylane task with db id '{db_id}'"
    )
    task_data: ProcessingTask = DB.session.execute(
        select(ProcessingTask).filter_by(id=db_id)
    ).scalar_one()

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    input_data_url: str = params.get("input_data", None)
    q_num: int = params.get("number_of_qubits", None)
    embedding_size: int = params.get("embedding_size", None)
    qnn_name: str = params.get("qnn_name", None)
    steps: int = params.get("training_steps", None)

    TASK_LOGGER.info(
        f"input_data: {input_data_url}, q_num: {q_num}, embedding_size: {embedding_size}, qnn_name: {qnn_name}, steps: {steps}"
    )

    if None in [input_data_url, q_num, embedding_size, qnn_name, steps]:
        raise ValueError("Request is missing one or more values.")

    with open_url(input_data_url, stream=True) as url_data:
        input_data_arr = np.genfromtxt(url_data.iter_lines(), delimiter=",")

    if input_data_arr.ndim == 1:
        input_data_arr = input_data_arr.reshape((1, -1))

    output_arr, model, c_optim, q_optim = simple_api.pennylane_hybrid_autoencoder(
        input_data_arr, q_num, embedding_size, qnn_name, steps
    )

    with SpooledTemporaryFile(mode="w") as output:
        np.savetxt(output, output_arr, delimiter=",")
        STORE.persist_task_result(
            db_id, output, "out.csv", "autoencoder-result", "text/csv"
        )
        output.seek(
            0
        )  # TODO remove separate output if task output is already persisted as file
        return "".join(output.readlines())
