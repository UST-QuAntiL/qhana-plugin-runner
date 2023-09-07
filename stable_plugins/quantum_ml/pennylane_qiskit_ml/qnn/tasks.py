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

# hybrid quantum neural network implemented based on:
# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
# https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html

import os
from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

from . import QNN
from .schemas import (
    InputParameters,
    QNNParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE

from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)

import numpy as np

# Timing tool
import time

# Scikit-learn tools
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from sklearn.metrics import accuracy_score

from .backend.load_utils import get_indices_and_point_arr, get_label_arr
from .backend.datasets import OneHotDataset
from .backend.train_and_test import train
from .backend.visualize import plot_data, plot_confusion_matrix


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{QNN.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    """
    train and test a classical or dressed quantum classification network
    """

    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------
    #        get input data
    # ------------------------------

    TASK_LOGGER.info(f"Starting new QNN calculation task with db id '{db_id}'")

    # load data from db
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = QNNParametersSchema().loads(task_data.parameters)

    train_points_url = input_params.train_points_url
    train_label_points_url = input_params.train_label_points_url
    test_points_url = input_params.test_points_url
    test_label_points_url = input_params.test_label_points_url
    n_qubits = input_params.n_qubits
    lr = input_params.lr
    batch_size = input_params.batch_size
    q_depth = input_params.q_depth
    network_enum = input_params.network_enum
    epochs = input_params.epochs
    shots = input_params.shots
    optimizer = input_params.optimizer
    weight_init = input_params.weight_init
    weights_to_wiggle = input_params.weights_to_wiggle
    diff_method = input_params.diff_method
    randomly_shuffle = input_params.randomly_shuffle
    visualize = input_params.visualize
    q_device_enum = input_params.device
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    if ibmq_token == "****":
        TASK_LOGGER.info(f"Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info(f"IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info(f"IBMQ_TOKEN environment variable not set")

    # load data
    train_id_list, train_data = get_indices_and_point_arr(train_points_url)
    train_labels, label_to_int, int_to_label = get_label_arr(
        train_label_points_url, train_id_list
    )

    test_id_list, test_data = get_indices_and_point_arr(test_points_url)
    test_labels = None
    if test_label_points_url != "" and test_label_points_url is not None:
        test_labels, label_to_int, int_to_label = get_label_arr(
            test_label_points_url,
            test_id_list,
            label_to_int=label_to_int,
            int_to_label=int_to_label,
        )
        test_labels = torch.from_numpy(test_labels)

    # ------------------------------
    #          new qnn
    # ------------------------------
    start_time = time.time()  # Start the computation timer

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

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

    preprocess_layers = [
        int(el) for el in input_params.preprocess_layers.split(",") if el != ""
    ]
    postprocess_layers = [
        int(el) for el in input_params.postprocess_layers.split(",") if el != ""
    ]
    hidden_layers = [int(el) for el in input_params.hidden_layers.split(",") if el != ""]

    model_parameters = dict(
        n_qubits=n_qubits,
        q_depth=q_depth,
        weight_init=weight_init,
        input_size=train_data.shape[1],
        output_size=n_classes,
        preprocess_layers=preprocess_layers,
        postprocess_layers=postprocess_layers,
        hidden_layers=hidden_layers,
        single_q_params=(weights_to_wiggle != 0),
        diff_method=diff_method,
    )

    if network_enum.needs_quantum_backend():
        model_parameters["quantum_device"] = q_device_enum.get_pennylane_backend(
            ibmq_token, custom_backend, n_qubits, shots
        )
    else:
        weights_to_wiggle = 0

    model = network_enum.get_neural_network(model_parameters)

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

    # Correct train labels
    train_labels = [int_to_label[el] for el in train_labels]

    # Plot title + confusion matrix
    plot_title = "Classification"
    conf_matrix = None
    if test_labels is not None:
        test_labels = [int_to_label[el] for el in test_labels]

        # Compute accuracy on test data
        test_accuracy = accuracy_score(test_labels, predictions)
        plot_title += f": accuracy on test data={test_accuracy}"

        # Create confusion matrix plot
        conf_matrix = plot_confusion_matrix(test_labels, predictions, int_to_label)

    info_str = f"_network_{network_enum.name}_optimizer_{optimizer.name}_epochs_{epochs}"

    # Output the data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(output_labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"labels{info_str}.json",
            "entity/label",
            "application/json",
        )

    if visualize:
        resolution = input_params.resolution
        fig = plot_data(
            train_data,
            train_id_list,
            train_labels,
            test_data,
            test_id_list,
            predictions,
            resolution,
            predictor=predictor,
            title=plot_title,
            label_to_int=label_to_int,
        )

        # show plot
        with SpooledTemporaryFile(mode="wt") as output:
            html = fig.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                f"classification_plot{info_str}.html",
                "plot",
                "text/html",
            )

    if conf_matrix is not None:
        with SpooledTemporaryFile(mode="wt") as output:
            html = conf_matrix.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                f"confusion_matrix{info_str}.html",
                "plot",
                "text/html",
            )

    # prepare weights output
    weights_dict = model.state_dict()
    for key, value in weights_dict.items():
        if isinstance(value, torch.Tensor):
            weights_dict[key] = value.tolist()
    weights_dict["net_type"] = str(model.__class__.__name__)

    # save weights in file
    with SpooledTemporaryFile(mode="w") as output:
        save_entities([weights_dict], output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"qnn-weights{info_str}.json",
            "qnn-weights",
            "application/json",
        )

    # save quantum circuit as qasm file
    if network_enum.needs_quantum_backend():
        qasm_string = model.get_representative_circuit()
        with SpooledTemporaryFile(mode="w") as output:
            output.write(qasm_string)
            STORE.persist_task_result(
                db_id,
                output,
                f"representative-circuit{info_str}.qasm",
                "representative-circuit",
                "application/qasm",
            )

    # Print final time
    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = round(total_time - minutes * 60)

    return "Total time: " + str(minutes) + " min, " + str(seconds) + " seconds"
