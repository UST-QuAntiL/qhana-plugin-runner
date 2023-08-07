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

import os
from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

from . import QCNN

from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)
from qhana_plugin_runner.storage import STORE

import time
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score

from .backend.load_utils import get_ids_and_data_arr, get_label_arr
from .backend.datasets import OneHotDataset
from .backend.train_and_test import train
from .backend.visualize import plot_confusion_matrix


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{QCNN.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new quantum cnn calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    train_data_url = input_params.train_data_url
    train_label_url = input_params.train_label_url
    test_data_url = input_params.test_data_url
    test_label_url = input_params.test_label_url
    randomly_shuffle = input_params.randomly_shuffle
    epochs = input_params.epochs
    optimizer = input_params.optimizer
    lr = input_params.lr
    qcnn_enum = input_params.qcnn_enum
    num_layers = input_params.num_layers
    batch_size = input_params.batch_size
    weight_init = input_params.weight_init
    weights_to_wiggle = input_params.weights_to_wiggle
    diff_method = input_params.diff_method
    backend = input_params.backend
    shots = input_params.shots
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend

    if ibmq_token == "****":
        TASK_LOGGER.info("Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info("IBMQ_TOKEN environment variable not set")

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    # load data
    train_id_list, train_data = get_ids_and_data_arr(train_data_url)
    train_labels, label_to_int, int_to_label = get_label_arr(
        train_label_url, train_id_list
    )

    test_id_list, test_data = get_ids_and_data_arr(test_data_url)
    test_labels = None
    if test_label_url != "" and test_label_url is not None:
        test_labels, label_to_int, int_to_label = get_label_arr(
            test_label_url,
            test_id_list,
            label_to_int=label_to_int,
            int_to_label=int_to_label,
        )
        test_labels = torch.from_numpy(test_labels)

    # ------------------------------
    #          new qnn
    # ------------------------------
    start_time = time.time()  # Start the computation timer

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.from_numpy(train_labels)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Prep data
    n_classes = len(label_to_int)
    train_dataloader = DataLoader(
        OneHotDataset(train_data, train_labels, n_classes),
        batch_size=batch_size,
        shuffle=randomly_shuffle,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_qubits = qcnn_enum.get_qubit_need(train_data[0])

    q_parameters = dict(
        n_qubits=n_qubits,
        num_layers=num_layers,
        weight_init=weight_init,
        diff_method=diff_method,
        single_q_params=(weights_to_wiggle != 0),
    )

    q_parameters["quantum_device"] = backend.get_pennylane_backend(
        ibmq_token, custom_backend, n_qubits, shots
    )

    model = nn.Sequential()
    model.add_module("quantum_ccn", qcnn_enum.get_neural_network(q_parameters))
    model.add_module(
        "pooling1", nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
    )
    model.add_module("flatten1", nn.Flatten())
    model.add_module("linear1", nn.LazyLinear(64))
    model.add_module("act_func1", nn.ReLU())
    model.add_module("dropout1", nn.Dropout(0.4))
    model.add_module("output", nn.LazyLinear(n_classes))

    model = model.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # select optimizer
    opt = optimizer.get_optimizer(model, lr)

    # train network
    train(model, train_dataloader, loss_fn, opt, epochs, weights_to_wiggle)

    # test network
    def predictor(data: torch.Tensor) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return torch.argmax(model(data), dim=1).tolist()

    predictions = [int_to_label[el] for el in predictor(test_data)]

    # Prepare labels to be saved
    output_labels = []

    for ent_id, pred in zip(test_id_list, predictions):
        output_labels.append({"ID": ent_id, "href": "", "label": pred})

    # Confusion matrix
    test_accuracy = ""
    if test_labels is not None:
        test_labels = [int_to_label[el] for el in test_labels]

        # Compute accuracy on test data
        test_accuracy = f" accuracy: {accuracy_score(test_labels, predictions)}"

        # Create confusion matrix plot
        conf_matrix = plot_confusion_matrix(test_labels, predictions, int_to_label)

    # prepare weights output
    weights_dict = model.state_dict()
    for key, value in weights_dict.items():
        if isinstance(value, torch.Tensor):
            weights_dict[key] = value.tolist()
    weights_dict["net_type"] = str(model.__class__.__name__)

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(output_labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "labels.json",
            "entity/label",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        output.write(conf_matrix.to_html())
        STORE.persist_task_result(
            db_id,
            output,
            "confusion_matrix.html",
            "plot",
            "text/html",
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities([weights_dict], output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "qnn-weights.json",
            "qnn-weights",
            "application/json",
        )

    # save quantum circuit as qasm file
    qasm_string = model._modules["quantum_ccn"].get_representative_circuit(train_data[0])
    with SpooledTemporaryFile(mode="w") as output:
        output.write(qasm_string)
        STORE.persist_task_result(
            db_id,
            output,
            "representative-circuit.qasm",
            "representative-circuit",
            "application/qasm",
        )

    # Print final time
    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = round(total_time - minutes * 60)

    return (
        "Total time: "
        + str(minutes)
        + " min, "
        + str(seconds)
        + " seconds"
        + test_accuracy
    )
