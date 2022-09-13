# Support Vector Machines (classical and quantum)
from enum import Enum
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Optional, Mapping
from json import dumps
from xmlrpc.client import Boolean

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

        if feature_map_name == FeatureMap.z_feature_map:
            return ZFeatureMap(
                feature_dimension=feature_dimension,
                reps=reps,
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
        use_quantum=False,
        visualize=False,
        regularization_C=float,
        kernel=ClassicalKernelEnum,
        degree=int,
        backend=QuantumBackends.aer_statevector_simulator,
        ibmq_token=str,
        ibmq_custom_backend=str,
        feature_map=FeatureMap.z_feature_map,
        entanglement=Entanglement.linear,
        reps=int,
        shots=int,
    ):
        self.entity_points_url = entity_points_url
        self.clusters_url = clusters_url
        self.use_quantum = use_quantum
        self.visualize = visualize
        self.regularization_C = regularization_C
        self.kernel = kernel
        self.degree = degree
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.ibmq_custom_backen = ibmq_custom_backend
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
    visualize = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "visualize classification",
            "description": "plot the decision boundary and the support vectors for the trained classifier",
            "input_type": "checkbox",
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
    ibmq_custom_backend = ma.fields.String(
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
            "description": "Specifies the entanglement structure. (For feature map. not for ZFeatureMap)",
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
            schema.fields["visualize"].data_key: False,
            schema.fields["regularization_C"].data_key: 1.0,
            schema.fields["kernel"].data_key: ClassicalKernelEnum.rbf,
            schema.fields["degree"].data_key: 3,
            schema.fields["backend"].data_key: QuantumBackends.aer_statevector_simulator,
            schema.fields["ibmq_token"].data_key: "",
            schema.fields["ibmq_custom_backend"].data_key: "",
            schema.fields["feature_map"].data_key: FeatureMap.z_feature_map,
            schema.fields["entanglement"].data_key: Entanglement.linear,
            schema.fields["reps"].data_key: 2,
            schema.fields["shots"].data_key: 1024,
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
        return "qiskit==0.27\nqiskit-aer~=0.8.2\nscikit-learn~=0.24.2\nqiskit-machine-learning"


TASK_LOGGER = get_task_logger(__name__)


def get_classical_SVC(data, labels, c=1.0, kernel="rbf", degree=3):  # TODO kernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    TASK_LOGGER.info("classical supportvector machine")

    csvc = SVC(C=c, kernel=kernel, degree=degree)
    csvc.fit(data, labels)

    return csvc


def get_quantum_SVC(data, labels, input_params):
    # from qiskit import Aer
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit.utils import QuantumInstance
    from qiskit_machine_learning.kernels.quantum_kernel import QuantumKernel

    TASK_LOGGER.info("quantum supportvector machine")

    backend = input_params.backend
    TASK_LOGGER.info(f"Loaded input parameters from db: backend='{backend}'")
    ibmq_token = input_params.ibmq_token
    TASK_LOGGER.info(f"Loaded input parameters from db: ibmq_token='{ibmq_token}'")
    ibmq_custom_backend = input_params.ibmq_custom_backen
    TASK_LOGGER.info(
        f"Loaded input parameters from db: ibmq_custom_backend='{ibmq_custom_backend}'"
    )

    # get quantum backend
    backend_device = QuantumBackends.get_quantum_backend(
        backendEnum=backend,
        ibmqToken=ibmq_token,
        customBackendName=ibmq_custom_backend,
    )

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

    quantum_instance = QuantumInstance(
        backend_device, seed_simulator=9283712, seed_transpiler=9283712, shots=shots
    )
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    qsvc = QSVC(quantum_kernel=qkernel)
    qsvc.fit(data, labels)
    return qsvc


def get_visualization(svm, train_data, test_data, train_labels, test_labels, score):
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
    factor = 2  # TODO
    x_values = np.append(train_x, test_x)
    y_values = np.append(train_y, test_y)

    x_min = np.min(x_values) * factor
    x_max = np.max(x_values) * factor
    y_min = np.min(y_values) * factor
    y_max = np.max(y_values) * factor
    print(x_min, x_max, y_min, y_max)

    # generate gridpoints
    h = 0.1  # how fine grained the background contour should be
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
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
    # print("SUPPORT VECTORS", support)
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
    entity_points_url = input_params.entity_points_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entity_points_url='{entity_points_url}'"
    )
    clusters_url = input_params.clusters_url
    TASK_LOGGER.info(f"Loaded input parameters from db: clusters_url='{clusters_url}'")

    use_default_data = False  # TODO set parameter in GUI
    if use_default_data:
        points, labels = twospirals(20, turns=1.52)  # qsvm aer_statevector very slow!
    else:
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
        figure_main = get_visualization(
            svm, train_data, test_data, train_labels, test_labels, score
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


# TODO quantum SVM supportvector list is empty!! why?
# TODO hide GUI elements when irrelevant

# (TODO neues plugin für qnn/nn für classification)

# TODO (neues plugin mit Pegasos QSVC könnte interessant sein)
# https://qiskit.org/documentation/machine-learning/tutorials/07_pegasos_qsvc.html


# TODO read? https://medium.com/qiskit/training-quantum-kernels-for-machine-learning-using-qiskit-617f6e4ed9ac


# https://medium.com/@patrick.huembeli/introduction-into-quantum-support-vector-machines-727f3ccfa2b4
# ----------------------------------------------------
# supervised classification on a quantum computer
# ----------------------------------------------------
#   need:
#       datapoint X. translated into quantum datapoint by a circuit V
#       parameterised circuit W to process data (parameters theta)
#       measure -> return classification value

# => ansatz: W(theta)V(phi(x))|0>
# = quantum variational circuit

# ------
# QSVM
# ------

# quantum feature maps V(phi(x)) to translate classical data x into quantum states
# build the kernel of the SVM out of these quantum states

# calculate kernel matrix on the quantum computer
# then train QSVM the same way as classical SVM
# (there are also QSVMs where also training on quantum computer.. not useful yet)


# Define quantum kernel
#   inner product of the (quantum) feature maps 𝐾(𝑥⃗,𝑧⃗)=|⟨Φ(𝑥⃗)|Φ(𝑧⃗)⟩|²
#   (idea: choose a quantum feature map that is not easy to simulate with classical computers => quantum advantage?)

# Feature map
#   ansatz: V(Φ(𝑥⃗)) = U(Φ(𝑥⃗))⊗𝐻ⁿ
#       (simplifies a lot when we only let two qubits interact at a time
#           => interactions ZZ and non interacting terms Z)
#       𝐻ⁿ is the Hadamard gate applied to each qubits (n = number of qubits)
#   classical functions:  Φᵤ(𝑥⃗) = xᵤ and Φᵤᵥ(𝑥⃗) = (π-xᵤ)(π-xᵥ).
#   ansatz for n=2 qubits and classical data vector 𝑥⃗= (x₁, x₂):
#       feature map: U(Φ(𝑥⃗)) = exp(i {x₁ Z₁ + x₂ Z₂ + (π - x₁)( π - x₂) Z₁ Z₂})
#   depth of the circuits (depth 2 means we repeat the ansatz two times: V(Φ(𝑥⃗)) = U(Φ(𝑥⃗))⊗𝐻ⁿ⊗U(Φ(𝑥⃗))⊗𝐻ⁿ)

# Measuring the Quantum Kernel
#   extract info about quantum kernel from quantum circuit and feed into classical SVM algorithm
#   we want to measure overlap f 2 states 𝐾(𝑥⃗,𝑧⃗)=|⟨Φ(𝑥⃗)|Φ(𝑧⃗)⟩|²
#   use specific circuit and
#   measure ourput, frequency of the measurement string [0,0,0,0,...] gives estimate of overlap  |⟨Φ(𝑥⃗)|Φ(𝑧⃗)⟩|²

# Build qiskit circuit for feature map
#   qiskit aqua:
#       pre-built function to construct quantum circuit for V(Φ(𝑥⃗))
#       second order expansion = the number of interacting qubits
#       function SecondOrderExpansion:
#           arguments:
#               feature_dimension (dimension of the input data  𝑥⃗ == number of qubits n)
#               depth (number of repetitions of the feature map)

# QSVM Algorithm in QISKIT
#   qiskit aqua provides pre-defined function to train the whole QSVM
#       we have to provide: featuremap, training and test set
#       similar to classical SVM except but Kernels come from a quantum distribution
#   minimizes the loss 𝐿(𝑊)=∑ᵤ𝑤ᵤ−½ ∑ᵤᵥ𝑦ᵤ𝑦ᵥαᵤαᵥ 𝐾(𝑥⃗ᵤ,𝑥⃗ᵥ) by optimizing parameters W
#   after training we can predict a label 𝑦′ of a data instance 𝑠⃗ with 𝑦′=sign(∑ᵤ𝑦ᵤαᵤ𝐾(𝑥⃗ᵤ,𝑠⃗)).

# from qiskit.aqua.algorithms import QSVM
# from qiskit.aqua import run_algorithm, QuantumInstance
# from qiskit import BasicAer

# qsvm = QSVM(feature_map, training_input, test_input)
# backend = BasicAer.get_backend('qasm_simulator')
# quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=10598, seed_transpiler=10598)
# result = qsvm.run(quantum_instance)
# print("test accuracy", result['testing_accuracy'])
# y_test = qsvm.predict(test_set, quantum_instance)
