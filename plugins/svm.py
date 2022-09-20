# Support Vector Machines (classical and quantum)
from enum import Enum
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Optional, Mapping
import os

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

from qiskit import IBMQ
from qiskit import Aer
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap

import numpy as np  # TODO

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


class FeatureMap(Enum):
    z_feature_map = "ZFeatureMap"
    zz_feature_map = "ZZFeatureMap"
    pauli_feature_map = "PauliFeatureMap"

    @staticmethod
    def get_feature_map(feature_map_name, feature_dimension, reps, entanglement):
        """
        select and return a FeatureMap based on the given parameters

        feature_map_name: type of feature map (FeatureMap enum)
        feature_dimension: numbre of qubits in the circuit (int)
        reps: the number of repeated circuits (int)
        entanglement: specifies the entanglement structure (string)
        """

        if feature_map_name == FeatureMap.z_feature_map:
            return ZFeatureMap(
                feature_dimension=feature_dimension, reps=reps
            )  # no entanglement parameter
        elif feature_map_name == FeatureMap.zz_feature_map:
            return ZZFeatureMap(
                feature_dimension=feature_dimension, entanglement=entanglement, reps=reps
            )
        elif feature_map_name == FeatureMap.pauli_feature_map:
            return PauliFeatureMap(
                feature_dimension=feature_dimension, entanglement=entanglement, reps=reps
            )
        else:
            TASK_LOGGER.error(
                "No such feature map available: {}".format(feature_map_name)
            )


class Entanglement(Enum):
    linear = "linear"
    full = "full"


class QuantumBackends(Enum):
    # from https://github.com/UST-QuAntiL/qhana/blob/main/qhana/backend/clustering.py
    custom_ibmq = "custom_ibmq"
    aer_statevector_simulator = "aer_statevector_simulator"
    aer_qasm_simulator = "aer_qasm_simulator"
    ibmq_qasm_simulator = "ibmq_qasm_simulator"
    ibmq_16_melbourne = "ibmq_16_melbourne"
    ibmq_armonk = "ibmq_armonk"
    ibmq_5_yorktown = "ibmq_5_yorktown"
    ibmq_ourense = "ibmq_ourense"
    ibmq_vigo = "ibmq_vigo"
    ibmq_valencia = "ibmq_valencia"
    ibmq_athens = "ibmq_athens"
    ibmq_santiago = "ibmq_santiago"

    @staticmethod
    def get_quantum_backend(backendEnum, ibmqToken=None, customBackendName=None):
        """
        select and return a quantum backend based on the given parameters

        backendEnum: quantum backend type to selct (QuantumBackends Enum)
        ibmqToken:
        customBackendName:
        """
        backend = None
        if backendEnum.name.startswith("aer"):
            # Use local AER backend
            aerBackendName = backendEnum.name[4:]
            backend = Aer.get_backend(aerBackendName)
        elif backendEnum.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmqToken)
            backend = provider.get_backend(backendEnum.name)
        elif backendEnum.name.startswith("custom_ibmq"):
            provider = IBMQ.enable_account(ibmqToken)
            backend = provider.get_backend(customBackendName)
        else:
            TASK_LOGGER.error("Unknown quantum backend specified!")
        return backend


class InputParameters:
    def __init__(
        self,
        entity_points_url: str,
        clusters_url: str,
        resolution: int,
        regularization_C: float,
        kernel: ClassicalKernelEnum,
        degree: int,
        ibmq_token: str,
        custom_backend: str,
        reps: int,
        shots: int,
        backend: QuantumBackends,
        feature_map: FeatureMap,
        entanglement: Entanglement,
        use_quantum=False,
        use_default_data=False,
        visualize=False,
    ):
        self.entity_points_url = entity_points_url
        self.clusters_url = clusters_url
        self.use_quantum = use_quantum
        self.use_default_data = use_default_data
        self.visualize = visualize
        self.resolution = resolution
        self.regularization_C = regularization_C
        self.kernel = kernel
        self.degree = degree
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.custom_backend = custom_backend
        self.feature_map = feature_map
        self.entanglement = entanglement
        self.reps = reps
        self.shots = shots

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
    use_default_data = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Use default dataset",
            "description": "Use internally generated dataset (no input files required).",
            "input_type": "checkbox",
        },
    )
    entity_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Data points URL",
            "description": "URL to a json file with the data points.",
            "input_type": "text",
        },
    )
    clusters_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="clusters",
        data_content_types="application/json",
        metadata={
            "label": "Labels URL",
            "description": "URL to a json file with the labels.",
            "input_type": "text",
        },
    )
    use_quantum = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Use quantum",
            "description": "Whether to use quantum svm or classical svm.",
            "input_type": "checkbox",
        },
    )
    visualize = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Visualize classification",
            "description": "Plot the decision boundary and the support vectors for the trained classifier.",
            "input_type": "checkbox",
        },
    )
    resolution = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Resolution",
            "description": "Resolution of the visualization. How finegrained the evaluation of the classifier should be.",
            "input_type": "text",
        },
    )
    backend = EnumField(
        QuantumBackends,
        required=True,
        allow_none=False,
        metadata={
            "label": "Backend",
            "description": "QC or simulator that will be used.",
            "input_type": "select",
        },
    )
    ibmq_token = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "IBMQ Token",
            "description": "Token for IBMQ.",
            "input_type": "text",
        },
    )
    custom_backend = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "Custom backend",
            "description": "Custom backend for IBMQ.",
            "input_type": "text",
        },
    )
    feature_map = EnumField(
        FeatureMap,
        required=True,
        allow_none=False,
        metadata={
            "label": "Featuremap",
            "description": "Feature map module used to transform data.",
            "input_type": "select",
        },
    )
    entanglement = EnumField(
        Entanglement,
        required=True,
        allow_none=False,
        metadata={
            "label": "Entanglement",
            "description": "Specifies the entanglement structure. (For feature map. Except for ZFeatureMap)",
            "input_type": "select",
        },
    )
    reps = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Repetitions",
            "description": "Number of repreated circuits.",
            "input_type": "text",
        },
    )
    shots = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Shots",
            "description": "Number of repetitions of each circuits, for sampling.",
            "input_type": "text",
        },
    )

    regularization_C = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Regularization parameter C",
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
            schema.fields["use_default_data"].data_key: False,
            schema.fields["visualize"].data_key: False,
            schema.fields["resolution"].data_key: 80,
            schema.fields["regularization_C"].data_key: 1.0,
            schema.fields["kernel"].data_key: ClassicalKernelEnum.rbf.value,
            schema.fields["degree"].data_key: 3,
            schema.fields[
                "backend"
            ].data_key: QuantumBackends.aer_statevector_simulator.value,
            schema.fields["feature_map"].data_key: FeatureMap.z_feature_map.value,
            schema.fields["entanglement"].data_key: Entanglement.linear.value,
            schema.fields["reps"].data_key: 2,
            schema.fields["shots"].data_key: 1024,
        }

        if "IBMQ_BACKEND" in os.environ:
            default_values[schema.fields["backend"].data_key] = os.environ["IBMQ_BACKEND"]

        if "IBMQ_TOKEN" in os.environ:
            default_values[schema.fields["ibmq_token"].data_key] = "****"

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
    description = "Classifies data with a classical or quantum support vector machine"

    tags = ["classification"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return SVM_BLP

    def get_requirements(self) -> str:
        # return "qiskit==0.27\nqiskit-aer~=0.8.2\nscikit-learn~=0.24.2\nqiskit-machine-learning"
        return "qiskit==0.27\nqiskit-aer~=0.8.2\nscikit-learn~=1.1\nqiskit-machine-learning"  # qiskit-terra==0.17.4 (for qiskit==0.27) but qiskit-machine-learning needs newer version?


TASK_LOGGER = get_task_logger(__name__)


def get_classical_SVC(data, labels, c=1.0, kernel="rbf", degree=3):  # TODO kernel
    """
    train classical support vector classifier with the given parameters and data

    data: training data
    labels: training labels
    c: regularization parameter. The lower C the stronger the regularization (float)
    kernel: kernel type used for the support vector machine (string)
    degree: defree of the polynomial kernel function ('poly'). Ignored by all oterh kernels (int)
    """
    from sklearn.svm import SVC

    TASK_LOGGER.info("classical supportvector machine")

    csvc = SVC(C=c, kernel=kernel, degree=degree)
    csvc.fit(data, labels)

    return csvc


def get_quantum_SVC(data, labels, input_params):
    """
    train quantum support vector classifier with the given parameters and data

    data: training data
    labels: training labels
    input_params: containing quantum backend parameters, feature map parameters, shots
    """
    # from qiskit import Aer
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit.utils import QuantumInstance
    from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel

    TASK_LOGGER.info("quantum supportvector machine")

    # load quantum backend parameters
    backend = input_params.backend
    TASK_LOGGER.info(f"Loaded input parameters from db: backend='{backend}'")
    ibmq_token = input_params.ibmq_token
    TASK_LOGGER.info(f"Loaded input parameters from db: ibmq_token='{ibmq_token}'")
    custom_backend = input_params.custom_backend
    TASK_LOGGER.info(
        f"Loaded input parameters from db: custom_backend='{custom_backend}'"
    )

    # get quantum backend
    backend_device = QuantumBackends.get_quantum_backend(
        backendEnum=backend,
        ibmqToken=ibmq_token,
        customBackendName=custom_backend,
    )

    # load feature map parameters
    feature_map = input_params.feature_map
    TASK_LOGGER.info(f"Loaded input parameters from db: feature_map='{feature_map}'")
    entanglement = input_params.entanglement
    TASK_LOGGER.info(f"Loaded input parameters from db: entanglement='{entanglement}'")
    reps = input_params.reps
    TASK_LOGGER.info(f"Loaded input parameters from db: reps='{reps}'")

    dimension = len(data[0])

    # get feature map
    feature_map = FeatureMap.get_feature_map(
        feature_map_name=feature_map,
        feature_dimension=dimension,
        reps=reps,
        entanglement=entanglement.name,
    )

    shots = input_params.shots
    TASK_LOGGER.info(f"Loaded input parameters from db: shots='{shots}'")

    # get quantum instance
    quantum_instance = QuantumInstance(
        backend_device, seed_simulator=9283712, seed_transpiler=9283712, shots=shots
    )
    # get quantum kernel
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    # get and train quantum support vector machine
    qsvc = QSVC(quantum_kernel=qkernel)
    qsvc.fit(data, labels)
    return qsvc


def get_visualization(
    svm, train_data, test_data, train_labels, test_labels, score, resolution
):
    """
    plot the classification results for training and test data and furthermore draw the decision boundary between classes

    svm: classifying function
    train_data: data that was used during training
    test_data: data used to test accuracy of classifier
    train_labels: ground truth classification for train_data
    test_labels: ground truth classification for test_data
    score: classification accuracy value on the train data
    resolution: how many evaluations of the classifier in each dimension
    """
    # Matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.lines import Line2D

    # number of classes
    classes = list(set(list(train_labels) + list(test_labels)))
    n_classes = len(classes)

    # color map for scatter plots
    colors = cm.get_cmap("rainbow", n_classes)

    # figure and subplot
    fig, ax = plt.subplots()

    # draw decision boundaries

    train_x = [element[0] for element in train_data]
    train_y = [element[1] for element in train_data]

    test_x = [element[0] for element in test_data]
    test_y = [element[1] for element in test_data]

    # bounds of the figure and grid
    factor = 2  # TODO # to create margin between data points and border of the plot
    x_values = np.append(train_x, test_x)
    y_values = np.append(train_y, test_y)

    x_min = np.min(x_values) * factor
    x_max = np.max(x_values) * factor
    y_min = np.min(y_values) * factor
    y_max = np.max(y_values) * factor
    TASK_LOGGER.info(x_min, x_max, y_min, y_max)

    # generate gridpoints

    # how fine grained the background contour should be
    x_range = x_max - x_min
    y_range = y_max - y_min
    hx = x_range / resolution
    hy = y_range / resolution
    TASK_LOGGER.info("how fine grained?", hx, hy)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # calculate class predictions for gridpoints
    grid_results = svm.predict(grid_points)
    Z = grid_results.reshape(xx.shape)

    # draw class predictions for grid as background
    ax.contourf(
        xx, yy, Z, levels=n_classes - 1, linestyles=["-"], cmap="winter", alpha=0.3
    )

    # draw training data
    ax.scatter(
        train_x,
        train_y,
        c=train_labels,
        s=100,
        lw=0,
        vmin=0,
        vmax=n_classes,
        cmap=colors,
    )
    # mark train data
    ax.scatter(train_x, train_y, c="b", s=50, marker="x")

    # scatter test set
    ax.scatter(
        test_x,
        test_y,
        c=test_labels,
        s=100,
        lw=0,
        vmin=0,
        vmax=n_classes,
        cmap=colors,
    )

    support = svm.support_vectors_

    # mark support vectors
    supp_x = [x[0] for x in support]
    supp_y = [x[1] for x in support]
    ax.scatter(supp_x, supp_y, s=150, linewidth=0.5, facecolors="none", edgecolors="g")

    # create legend elements

    # legend elements for classes
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=i,
            markerfacecolor=colors(i),
            markersize=10,
            ls="",
        )
        for i in range(n_classes)
    ]
    # legend element for training element crosses
    legend_elements = legend_elements + [
        Line2D(
            [0],
            [0],
            marker="x",
            color="b",
            label="train data",
            markerfacecolor="g",
            markersize=8,
            ls="",
        )
    ]

    # legend element for support vector circles
    legend_elements = legend_elements + [
        Line2D(
            [0],
            [0],
            marker="o",
            label="support vectors",
            markerfacecolor="none",
            markeredgecolor="g",
            markersize=8,
            ls="",
        )
    ]

    # add legend elements to plot
    ax.legend(handles=legend_elements, loc="upper left")

    # set pot title
    ax.set_title("Classification \naccuracy on test data={}".format(score))

    return fig


# ------------------------------
#       dataset generation
# ------------------------------
def twospirals(n_points, noise=0.7, turns=1.52):
    """Returns the two spirals dataset."""
    n = np.sqrt(np.random.rand(n_points, 1)) * turns * (2 * np.pi)
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points).astype(int), np.ones(n_points).astype(int))),
    )


@CELERY.task(name=f"{SVM.instance.identifier}.demo_task", bind=True)
def demo_task(self, db_id: int) -> str:
    """
    train and test a classical or quantum support vector classifier
    """
    import base64
    from io import BytesIO

    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = SVMSchema().loads(task_data.parameters)

    use_quantum = input_params.use_quantum
    TASK_LOGGER.info(f"Loaded input parameters from db: use_quantum='{use_quantum}'")
    visualize = input_params.visualize
    TASK_LOGGER.info(f"Loaded input parameters from db: visualize='{visualize}'")
    use_default_data = input_params.use_default_data
    TASK_LOGGER.info(
        f"Loaded input parameters from db: use_default_data='{use_default_data}'"
    )

    if use_default_data:
        points, labels = twospirals(50, turns=0.52)
        # points, labels = twospirals(50, turns=1.52)
        # shuffle (so that train and test data will both have all kinds of data)

        from sklearn.preprocessing import StandardScaler

        # points = StandardScaler().fit_transform(points)  # TODO remove?

        indices = np.arange(len(labels))
        np.random.shuffle(indices)  # TODO Gui element: shuffle?
        points = points[indices]
        labels = labels[indices]
    else:
        entity_points_url = input_params.entity_points_url
        TASK_LOGGER.info(
            f"Loaded input parameters from db: entity_points_url='{entity_points_url}'"
        )
        clusters_url = input_params.clusters_url
        TASK_LOGGER.info(
            f"Loaded input parameters from db: clusters_url='{clusters_url}'"
        )
        # load data from file
        entity_points = open_url(entity_points_url).json()
        clusters_entities = open_url(clusters_url).json()

        # get lists of data
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

        svm = get_quantum_SVC(train_data, train_labels, input_params)
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
    # instead of predictions and accuracy:
    score = svm.score(test_data, test_labels)

    print("ACCURACY", accuracy, "SCORE", score)

    if visualize:
        # visualize classification
        resolution = input_params.resolution
        TASK_LOGGER.info(f"Loaded input parameters from db: resolution='{resolution}'")

        figure_main = get_visualization(
            svm, train_data, test_data, train_labels, test_labels, score, resolution
        )

        # plot to html
        tmpfile = BytesIO()
        figure_main.savefig(tmpfile, format="png")
        encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        html = "<img src='data:image/png;base64,{}'>".format(encoded)

        # show plot
        with SpooledTemporaryFile(mode="wt") as output:
            output.write(html)
            STORE.persist_task_result(
                db_id,
                output,
                "plot.html",
                "plot",
                "text/html",
            )

    # save support vectors in a file
    support_vectors = []  # TODO better solution? each support vector individually?
    support_vectors.append({"support_vectors": svm.support_vectors_.tolist()})

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(support_vectors, output, "application/json")
        STORE.persist_task_result(
            db_id, output, "support-vectors.json", "support-vectors", "application/json"
        )

    return "DONE with accuracy: " + str(accuracy)


# TODO quantum SVM supportvector list is empty!! why? not supposed to be used?
# TODO hide GUI elements when irrelevant


# (TODO neues plugin für qnn/nn für classification)
# TODO (neues plugin mit Pegasos QSVC könnte interessant sein)
# https://qiskit.org/documentation/machine-learning/tutorials/07_pegasos_qsvc.html
