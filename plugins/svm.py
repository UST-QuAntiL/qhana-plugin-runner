# Support Vector Machines (classical and quantum)
from enum import Enum
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
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

from qhana_plugin_runner.api import EnumField
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
    FileUrl,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url

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


class ClassicalKernelEnum(Enum):
    rbf = "rbf"
    linear = "linear"
    poly = "poly"
    sigmoid = "sigmoid"


class InputParameters:
    def __init__(
        self,
        entity_points_url: str,
        clusters_url: str,
        use_quantum=False,
        regularization_C=float,
        kernel=ClassicalKernelEnum,
        degree=int,
    ):
        self.entity_points_url = entity_points_url
        self.clusters_url = clusters_url
        self.use_quantum = use_quantum
        self.regularization_C = regularization_C
        self.kernel = kernel
        self.degree = degree

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


class SVMSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Entity points URL",
            "description": "URL to a json file with the entity points.",
            "input_type": "text",
        },
    )
    clusters_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="clusters",
        data_content_types="application/json",
        metadata={
            "label": "Clusters URL",
            "description": "URL to a json file with the clusters.",
            "input_type": "text",
        },
    )
    use_quantum = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "use quantum",
            "description": "whether to use quantum svm or classical svm",
            "input_type": "checkbox",
        },
    )
    regularization_C = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "regularization parameter C",
            "description": "The strength of the regularization is inversely proportional to C. Must be strictly positive, the penalty is a squared l2 penalty.",
            "input_type": "text",
        },
    )
    kernel = EnumField(
        ClassicalKernelEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Kernel",
            "description": "Type of kernel to use for classical SVM.",
            "input_type": "select",
        },
    )
    degree = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Degree",
            "description": "Degree of the polynomial kernel function (poly). Ignored by all other kernels.",
            "input_type": "text",
        },
    )

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
            schema.fields["use_quantum"].data_key: False,
            schema.fields["regularization_C"].data_key: 1.0,
            schema.fields["kernel"].data_key: ClassicalKernelEnum.rbf,
            schema.fields["degree"].data_key: 3,
        }

        print("datadict before", data_dict)
        # overwrite default values with other values
        default_values.update(data_dict)
        data_dict = default_values

        print("datadict after", data_dict)

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


def get_classical_SVC(data, labels, c=1.0, kernel="rbf", degree=3):  # TODO kernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    TASK_LOGGER.info("classical supportvector machine")

    csvc = SVC(C=c, kernel=kernel, degree=degree)
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
    entity_points_url = input_params.entity_points_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entity_points_url='{entity_points_url}'"
    )
    clusters_url = input_params.clusters_url
    TASK_LOGGER.info(f"Loaded input parameters from db: clusters_url='{clusters_url}'")

    # load data from file
    entity_points = open_url(entity_points_url).json()
    clusters_entities = open_url(clusters_url).json()

    # get data
    clusters = {}
    for ent in clusters_entities:
        clusters[ent["ID"]] = ent["cluster"]
    points = []
    labels = []
    for ent in entity_points:
        points.append(ent["point"])
        labels.append(clusters[ent["ID"]])

    # TODO randomly shuffle data?

    # split training and test data
    test_percentage = 0.2
    n_data = len(labels)
    n_test = int(n_data * test_percentage)  # number of test data elements
    if n_test < 1:
        n_test = 1
    if n_test > n_data - 1:
        n_test = n_data - 1
    n_train = n_data - n_test  # Number of training points
    TASK_LOGGER.info(
        f"Number of data elements: n_train = '{n_train}', n_test = '{n_test}'"
    )

    train_data = points[:-n_test]
    test_data = points[-n_test:]

    train_labels = labels[:-n_test]
    test_labels = labels[-n_test:]

    # Support vector machine
    svm = None
    if use_quantum:

        svm = get_quantum_SVC(train_data, train_labels)
    else:
        c = input_params.regularization_C
        TASK_LOGGER.info(f"Loaded input parameters from db: c='{c}'")
        kernel = input_params.kernel.value
        TASK_LOGGER.info(f"Loaded input parameters from db: kernel='{kernel}'")
        degree = input_params.degree
        TASK_LOGGER.info(f"Loaded input parameters from db: degree='{degree}'")

        svm = get_classical_SVC(
            train_data, train_labels, c=c, kernel=kernel, degree=degree
        )

    # Test SVM
    predictions = svm.predict(test_data)
    # accuracy
    accuracy = np.sum(predictions == test_labels) / len(test_labels)

    # TODO visualize

    # save support vectors in a file
    support_vectors = []  # TODO better solution? each support vector individually?
    support_vectors.append({"support_vectors": svm.support_vectors_.tolist()})

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(support_vectors, output, "application/json")
        STORE.persist_task_result(
            db_id, output, "support-vectors.json", "support-vectors", "application/json"
        )

    return "DONE with accuracy: " + str(accuracy)


# TODO quantum SVM supportvector list is empty!! why?
# TODO visualize?
# TODO quantum SVM parameters in GUI
# TODO hide GUI elements when irrelevant
