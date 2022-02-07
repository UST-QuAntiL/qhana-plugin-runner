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

from enum import Enum
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE, post_load

from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "qnn"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


QNN_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="QNN plugin API",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


##### ????????????????
class InputParameters:
    def __init__(self, theta: float):
        self.theta = theta


class QNNParametersSchema(FrontendFormBaseSchema):
    theta = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Theta",
            "description": "The input parameter for the QNN (rotation parameter)",
            "input_type": "text",
        },
    )

    # ?????????????????
    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@QNN_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QNN_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @QNN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """QNN endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Quantum Neutral Network (QNN)",
            description="Simple (hardcoded) QNN",
            name=QNN.instance.identifier,
            version=QNN.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{QNN_BLP.name}.ProcessView"),
                ui_href=url_for(f"{QNN_BLP.name}.MicroFrontend"),
                data_input=[],  # TODO ?
                data_output=[
                    """DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )"""
                ],
            ),
            tags=[],  # TODO?
        )


@QNN_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the QNN plugin."""

    example_inputs = {  # TODO?
        "inputStr": "Sample input string.",
    }

    @QNN_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the QNN plugin.")
    @QNN_BLP.arguments(
        QNNParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QNN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @QNN_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the QNN plugin.")
    @QNN_BLP.arguments(
        QNNParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QNN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = QNNParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=QNN.instance.name,
                version=QNN.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{QNN_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{QNN_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )
        """
        data_dict = dict(data)
        fields = QNNParametersSchema().fields

        # define default values
        default_values = {
            fields["theta"].data_key: 0.0,
            fields["solver"].data_key: SolverEnum.auto,
            fields["scale"].data_key: False,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=QNN.instance.name,
                version=QNN.instance.version,
                schema=QNNParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{QNN_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{QNN_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )
        """


@QNN_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @QNN_BLP.arguments(QNNParametersSchema(unknown=EXCLUDE), location="form")
    @QNN_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @QNN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=QNNParametersSchema().dumps(arguments),
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


class QNN(QHAnaPluginBase):
    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QNN_BLP

    # def get_requirements(self) -> str:
    #    return "scikit-learn~=0.24.2" # TODO?


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{QNN.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new QNN calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = QNNParametersSchema().loads(task_data.parameters)
    theta = input_params.theta
    TASK_LOGGER.info(f"Loaded input parameters from db: theta='{theta}'")
    if theta is None:
        raise ValueError("No input argument provided!")
    """if theta:
        out_str = theta
        with SpooledTemporaryFile(mode="w") as output:
            output.write(theta)
            STORE.persist_task_result(
                db_id, output, "output.txt", "qnn-output", "text/plain"
            )
        return "result: " + repr(out_str)
    return "Empty input, no output could be generated!"
    """
    return testQuantumCircuitClass(theta)
    # return "Task is done"


##########################################################
# qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html
import numpy as np

# import matplotlib.pyplot as plt

# ModuleNotFoundError: No module named 'torch'
# import torch
# from torch.autograd import Function
# from torchvision import datasets, transforms
# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *

# create a quantum class with qiskit
#   here: a 1-qubit circuit with one trainable quantum parameter theta
#   for simplicity: hardcoded circuit, RY-rotation by theta
#             _____     _____________        _________
#   |0>-------| H |-----| RY(theta) |--------|measure|
#             _____     _____________        _________

#   measure in z-basis and calculate the oz expectation
class QuantumCircuit:
    """
    # This class provides a simple interface for interaction with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter("theta")  # input?

        self._circuit.h(all_qubits)  # initialize qubits with Hadamard gates
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)  # encode inputdata with RY rotation

        self._circuit.measure_all()  # measure all qubits
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit, self.backend)
        qobj = assemble(
            t_qc,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        # run
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


# test implementation
def testQuantumCircuitClass(theta):
    simulator = qiskit.Aer.get_backend("aer_simulator")

    circuit = QuantumCircuit(1, simulator, 100)
    # print("Expected value for rotation pi {}".format(circuit.run([np.pi])[0]))
    # circuit._circuit.draw()

    # TODO actually use theta
    return "Expected value for rotation pi %s (with given theta %s)" % (
        (circuit.run([theta])[0]),
        theta,
    )


############################################################

# quantum classical class with pytorch


# specify parameters
#   dataelements to classify
#


# quantum neural network
#   Initialize (balanced superposition)
#   Encode data (rotation)
#   Successively apply sequence of trainable rotation layers and constant entangling layers
#   measure local expectation value of the Z operator of each qubit -> classical output vector


# old qhana
#   QNN.py : classes
#       NNQuantumCircuit
#           - __init__
#           - run
#       HybridFunction
#           - forward
#           - backward
#       Hybrid
#           - __init__
#           - update_epoch
#           - init_weights
#           - forward
#       DressedQNN
#           - __init__
#           - update_epoch
#           - forward
#   QNNCircuitGenerator.py
#       QNNCircuitGenerator
#           - genCircuit
#   QNNcircuitExecutor.py
#       CircuitExecutor
#           - runCircuit
#           - parametrization_from_parameter_names


# readme (old qhana)
# classification : once clusters have been created, we can perform some classification tasks. The following clustering algorithms are implemented in QHAna
#       ClassicalSklearnNN      Classical neural network method using SKLearn [22]
#       HybridQNN               Hybrid quantum-classical Neutal Network using PyTorch and Qiskit [23]


# TODO linter
# flake8
# einstellungen > settings: Python linting flake8 enabled []
# if enabled: errors in qnn.py?
# because imports should all be at the top and all variables should be used
