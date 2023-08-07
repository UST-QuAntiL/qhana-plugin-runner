# Copyright 2023 QHAna plugin runner contributors.
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

from tempfile import SpooledTemporaryFile

from celery.utils.log import get_task_logger
from sqlalchemy import select

from .schemas import (
    InputParameters,
    HybridAutoencoderPennylaneRequestSchema,
)

from . import HybridAutoencoderPlugin
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.storage import STORE

from typing import List
from torch import Tensor, tensor, float32, less_equal
from torch.nn import MSELoss

from .backend.load_utils import get_indices_and_point_arr


TASK_LOGGER = get_task_logger(__name__)


def prepare_data_for_output(id_list: list, data: List[List[float]]):
    return [
        {
            **{"ID": id_, "href": ""},
            **{f"dim{dim}": value for dim, value in enumerate(point)},
        }
        for id_, point in zip(id_list, data)
    ]


@CELERY.task(
    name=f"{HybridAutoencoderPlugin.instance.identifier}.pennylane_hybrid_autoencoder_task",
    bind=True,
)
def hybrid_autoencoder_pennylane_task(self, db_id: int) -> str:
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

    input_params: InputParameters = HybridAutoencoderPennylaneRequestSchema().loads(
        task_data.parameters
    )

    train_data_url = input_params.train_points_url
    test_data_url = input_params.test_points_url
    q_num = input_params.number_of_qubits
    embedding_size = input_params.embedding_size
    qnn_name = input_params.qnn_name.get_qnn()
    steps = input_params.training_steps
    q_device_enum = input_params.backend
    shots = input_params.shots
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend

    # Log information about the input parameters
    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    if None in [train_data_url, q_num, embedding_size, qnn_name, steps]:
        raise ValueError("Request is missing one or more values.")

    train_id_list, train_data = get_indices_and_point_arr(train_data_url)
    test_id_list, test_data = get_indices_and_point_arr(test_data_url)

    (
        embedded_train_data,
        model,
        c_optim,
        q_optim,
    ) = simple_api.pennylane_hybrid_autoencoder(
        train_data,
        q_num,
        embedding_size,
        qnn_name,
        steps,
        q_device_enum.get_pennylane_backend(ibmq_token, custom_backend, q_num, shots),
    )
    test_data = tensor(test_data, dtype=float32)
    embedded_test_data = model.embed(test_data)
    recovered_test_data = model.reconstruct(embedded_test_data)

    mse = MSELoss()(recovered_test_data, test_data)

    # prepare weights output
    weights_dict = model.state_dict()
    for key, value in weights_dict.items():
        if isinstance(value, Tensor):
            weights_dict[key] = value.tolist()
    weights_dict["net_type"] = str(qnn_name)

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(
            prepare_data_for_output(train_id_list, embedded_train_data.tolist()),
            output,
            "application/json",
        )
        STORE.persist_task_result(
            db_id,
            output,
            "training_data_embedding.json",
            "entity/vector",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(
            prepare_data_for_output(test_id_list, embedded_test_data.tolist()),
            output,
            "application/json",
        )
        STORE.persist_task_result(
            db_id,
            output,
            "test_data_embedding.json",
            "entity/vector",
            "application/json",
        )

    # save weights in file
    with SpooledTemporaryFile(mode="w") as output:
        save_entities([weights_dict], output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "autoencoder-weights.json",
            "qnn-weights",
            "application/json",
        )

    return f"The autoencoder achieved a mean squared error of {mse} on the test data."
