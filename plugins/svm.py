# Support Vector Machines (classical and quantum)
from http import HTTPStatus
from typing import Optional, Mapping
from json import dumps

import marshmallow as ma
from marshmallow import EXCLUDE, post_load
from flask import abort, redirect
from flask.app import Flask
from flask.views import MethodView
from flask.globals import request
from flask.helpers import url_for
from flask.wrappers import Response
from flask.templating import render_template
from celery.canvas import chain
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    EntryPoint,
)

from qhana_plugin_runner.api.util import (
    SecurityBlueprint,
    MaBaseSchema,
    FrontendFormBaseSchema,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier


_plugin_name = "svm"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

SVM_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Support Vector Machine API",
    # template_folder="hello_world_templates",
)


class DemoResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        # data_url: str,
        use_quantum=True,
    ):
        # self.data_url = data_url
        self.use_quantum = use_quantum


class SVMSchema(FrontendFormBaseSchema):
    use_quantum = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "use quantum",
            "description": "whether to use quantum svm or classical svm",
            "input_type": "checkbox",
        },
    )

    print("USE QUANTUM", use_quantum)

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@SVM_BLP.route("/")
class PluginView(MethodView):
    """Plugins collection resource."""

    @SVM_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @SVM_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = SVM.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=SVM_BLP.description,
            name=plugin.identifier,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{SVM_BLP.name}.ProcessView"),
                ui_href=url_for(f"{SVM_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )
                ],
            ),
            tags=[],
        )


@SVM_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the SVM plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @SVM_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the SVM plugin.")
    @SVM_BLP.arguments(
        SVMSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @SVM_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return micri frontend."""
        return self.render(request.args, errors)

    @SVM_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @SVM_BLP.arguments(
        SVMSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @SVM_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        plugin = SVM.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = SVMSchema()

        data_dict = dict(data)

        # define default values
        default_values = {
            schema.fields["use_quantum"].data_key: True,
        }

        # overwrite default values with other values
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{SVM_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{SVM_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@SVM_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @SVM_BLP.arguments(SVMSchema(unknown=EXCLUDE), location="form")
    @SVM_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @SVM_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name=demo_task.name, parameters=SVMSchema().dumps(arguments)
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = demo_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class SVM(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return SVM_BLP

    def get_requirements(self) -> str:
        return "qiskit~=0.27\nqiskit-aer~=0.10.4\nqiskit-machine-learning~=0.4.0\nscikit-learn~=0.24.2"  # sklearn and version like in other plugins


TASK_LOGGER = get_task_logger(__name__)


def get_classical_SVC(data, labels):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    TASK_LOGGER.info("classical supportvector machine")

    csvc = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    csvc.fit(data, labels)

    return csvc


def get_quantum_SVC(data, labels):
    # more parameters: backend, quantum kernel?, featuremap?

    from qiskit import Aer
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit.circuit.library import ZFeatureMap
    from qiskit.utils import QuantumInstance
    from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel

    TASK_LOGGER.info("quantum supportvector machine")

    backend = Aer.get_backend("qasm_simulator")
    feature_map = ZFeatureMap(feature_dimension=2, reps=2)
    quantum_instance = QuantumInstance(
        backend, seed_simulator=9283712, seed_transpiler=9283712, shots=1024
    )
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    qsvc = QSVC(quantum_kernel=qkernel)
    qsvc.fit(data, labels)
    return qsvc


@CELERY.task(name=f"{SVM.instance.identifier}.demo_task", bind=True)
def demo_task(self, db_id: int) -> str:
    import numpy as np  # TODO

    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = SVMSchema().loads(task_data.parameters)

    use_quantum = input_params.use_quantum
    TASK_LOGGER.info(f"Loaded input parameters from db: use_quantum='{use_quantum}'")
    # parameters
    # CSVM:
    #   regularization parameter C=1.0
    #   kernel = rbf (linear, poly, rbf, sigmoid)
    #   degree of the poly kernel = 3 (irrelevant for all other kernels)
    # QSVC: (qiskit)
    #   backend = QuantumBackends.aer_statevector_simulator
    #   Ibmq_token = ""
    #   Ibmq_custom_backend = ""
    #   Featuremap = "ZFeatureMap" (ZFeatureMap, ZZFeatureMap, PauliFeatureMap)
    #   Entanglement = "linear" (full, linear)
    #   Reps = 2
    #   Shots = 1024

    data = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    labels = np.array([1, 1, 2, 2])
    sample_test = [[-0.8, -1]]

    svm = None
    if use_quantum:
        svm = get_quantum_SVC(data, labels)
    else:
        svm = get_classical_SVC(data, labels)

    # TODO accuracy

    # TODO visualize

    return "DONE " + str(svm.predict(sample_test)) + str(use_quantum)


# TODO use_quantum is always True..
