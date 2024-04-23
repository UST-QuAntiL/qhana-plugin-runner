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

from typing import Optional, List

from celery.utils.log import get_task_logger

from . import SVM
import numpy as np

from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import retrieve_filename
from .backend.load_utils import (
    load_kernel_matrix,
    get_indices_and_point_arr,
    get_label_arr,
    get_id_list,
)
from .backend.svm import get_svc

from sklearn.metrics import accuracy_score
from .backend.visualize import plot_data, plot_confusion_matrix
import muid


TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


@CELERY.task(name=f"{SVM.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new svm calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    train_points_url = input_params.train_points_url
    train_label_points_url = input_params.train_label_points_url
    test_points_url = input_params.test_points_url
    test_label_points_url = input_params.test_label_points_url
    kernel_enum = input_params.kernel_enum
    train_kernel_url = input_params.train_kernel_url
    test_kernel_url = input_params.test_kernel_url
    regularization_C = input_params.regularization_C
    degree = input_params.degree
    data_maps_enum = input_params.data_maps_enum
    entanglement_pattern = input_params.entanglement_pattern
    paulis = input_params.paulis
    reps = input_params.reps
    shots = input_params.shots
    backend = input_params.backend
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend
    visualize = input_params.visualize
    resolution = input_params.resolution

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    # load data
    train_data, test_data = None, None
    train_kernel, test_kernel = None, None

    concat_filenames = ""
    if (
        train_points_url is not None
        and train_points_url != ""
        and test_points_url is not None
        and test_points_url != ""
    ):
        train_id_list, train_data = get_indices_and_point_arr(train_points_url)
        test_id_list, test_data = get_indices_and_point_arr(test_points_url)

        concat_filenames += retrieve_filename(train_points_url)
        concat_filenames += retrieve_filename(test_points_url)
    else:
        # Load kernels
        train_id_to_idx, _, train_kernel = load_kernel_matrix(train_kernel_url)
        train_id_to_idx, test_id_to_idx, test_kernel = load_kernel_matrix(test_kernel_url)
        # Get id lists
        train_id_list = get_id_list(train_id_to_idx)
        test_id_list = get_id_list(test_id_to_idx)

        concat_filenames += retrieve_filename(train_kernel_url)
        concat_filenames += retrieve_filename(test_kernel_url)

    # Load labels
    train_labels, label_to_int, int_to_label = get_label_arr(
        train_label_points_url, train_id_list
    )
    test_labels = None
    if test_label_points_url != "" and test_label_points_url is not None:
        test_labels, label_to_int, int_to_label = get_label_arr(
            test_label_points_url,
            test_id_list,
            label_to_int=label_to_int,
            int_to_label=int_to_label,
        )

    # Prepare additional parameters for the chosen kernel
    if not kernel_enum.is_classical():
        backend = backend.get_qiskit_backend(ibmq_token, custom_backend, shots)
        kernel_kwargs = dict(
            backend=backend,
            n_qubits=train_data.shape[1],
            paulis=paulis.replace(" ", "").split(","),
            reps=reps,
            entanglement_pattern=entanglement_pattern.get_pattern(),
            data_map_func=data_maps_enum.get_data_mapping(),
        )
    else:
        kernel_kwargs = dict()

    # Get trained Support Vector Classifier (SVC)
    svc = get_svc(
        train_data,
        train_labels,
        regularization_C,
        kernel_enum.get_kernel(**kernel_kwargs),
        degree,
        train_kernel,
    )

    def predictor(X: np.array | List[List[float]]) -> List[float]:
        return [int_to_label[el] for el in svc.predict(X)]

    predictions = (
        predictor(test_kernel)
        if kernel_enum.value == "precomputed"
        else predictor(test_data)
    )

    # Prepare labels to be saved
    output_labels = []
    for ent_id, pred in zip(test_id_list, predictions):
        output_labels.append({"ID": ent_id, "href": "", "label": pred})

    # Correct train_labels
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

    fig = None
    if visualize:
        fig = plot_data(
            train_data,
            train_id_list,
            train_labels,
            test_data,
            test_id_list,
            test_labels,
            resolution=resolution,
            predictor=predictor,
            title=plot_title,
            label_to_int=label_to_int,
            support_vectors=svc.support_,
        )

    # Prepare support vectors
    support_vectors = [
        {
            **{"ID": train_id_list[idx], "href": ""},
            **{f"dim{dim}": value for dim, value in enumerate(train_data[idx])},
        }
        for idx in svc.support_
    ]

    concat_filenames += retrieve_filename(train_label_points_url)
    concat_filenames += retrieve_filename(test_label_points_url)
    filename_hash = get_readable_hash(concat_filenames)

    kernel_name = str(kernel_enum.name).replace("kernel", "").strip("_")

    info_str = (
        f"_svm_kernel_{kernel_name}_regularization_{regularization_C}_{filename_hash}"
    )

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(output_labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"labels{info_str}.json",
            "entity/label",
            "application/json",
        )

    if fig is not None:
        with SpooledTemporaryFile(mode="w") as output:
            html = fig.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                f"plot{info_str}.html",
                "plot",
                "text/html",
            )

    if conf_matrix is not None:
        with SpooledTemporaryFile(mode="w") as output:
            html = conf_matrix.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                f"confusion_matrix{info_str}.html",
                "plot",
                "text/html",
            )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(support_vectors, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"support-vectors{info_str}.json",
            "support-vectors",
            "application/json",
        )

    return "Result stored in file"
