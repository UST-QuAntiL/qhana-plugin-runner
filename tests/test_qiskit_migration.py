# Copyright 2026 QHAna plugin runner contributors.
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

"""Tests for the behaviour that changed when porting the plugins to qiskit 2.

These tests exercise plugin code and are skipped when the optional plugin
dependencies are not installed in the current environment.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_PATH = Path(__file__).resolve().parent.parent

QISKIT_ML_PATH = REPOSITORY_PATH / "stable_plugins" / "quantum_ml" / "qiskit_ml"
QISKIT_EXECUTOR_PATH = REPOSITORY_PATH / "plugins" / "qiskit_executor"
PENNYLANE_ML_PATH = (
    REPOSITORY_PATH / "stable_plugins" / "quantum_ml" / "pennylane_qiskit_ml"
)

# an OpenQASM 3 program, it cannot be parsed by the OpenQASM 2 parser
QASM3_CIRCUIT = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

# the same circuit as OpenQASM 2, it cannot be parsed by the OpenQASM 3 importer
QASM2_CIRCUIT = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"""


def load_plugin_module(name: str, path: Path) -> ModuleType:
    """Import a plugin source file directly, without installing the plugin."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, (
        f"Could not build an import spec for '{path}'. The file was probably moved "
        f"or renamed and this test needs to be updated."
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def svm_kernel() -> ModuleType:
    """Return the kernel module of the qiskit svm plugin."""
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_machine_learning")
    return load_plugin_module(
        "svm_kernel", QISKIT_ML_PATH / "svm" / "backend" / "kernel.py"
    )


@pytest.fixture(scope="module")
def kernel_estimation_kernel() -> ModuleType:
    """Return the kernel module of the kernel estimation plugin."""
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_machine_learning")
    return load_plugin_module(
        "kernel_estimation_kernel",
        QISKIT_ML_PATH / "qiskit_quantum_kernel_estimation" / "backend" / "kernel.py",
    )


def test_qasm3_circuit_is_parsed_by_the_qasm3_importer():
    """The optional qasm3 importer is installed and parses OpenQASM 3."""
    pytest.importorskip("qiskit_qasm3_import")
    from qiskit.qasm3 import loads as loads3

    circuit = loads3(QASM3_CIRCUIT)

    assert circuit.num_qubits == 2, (
        "The OpenQASM 3 importer did not build the declared 2 qubit register. "
        "Either the importer package is missing or its output changed."
    )


def test_qasm2_circuit_is_rejected_by_the_qasm3_importer():
    """An OpenQASM 2 program raises the error the executors fall back on.

    The executors catch exactly this exception to retry with the OpenQASM 2
    parser, so widening or narrowing it would break the fallback.
    """
    pytest.importorskip("qiskit_qasm3_import")
    from qiskit.qasm3 import QASM3ImporterError
    from qiskit.qasm3 import loads as loads3

    with pytest.raises(QASM3ImporterError):
        loads3(QASM2_CIRCUIT)


def test_qasm3_importer_error_is_a_qiskit_error():
    """QASM3ImporterError must stay narrower than QiskitError.

    The executors catch QASM3ImporterError alone. Catching QiskitError instead
    would also swallow the MissingOptionalLibraryError raised when the importer
    package is absent, silently downgrading OpenQASM 3 input to OpenQASM 2.
    """
    from qiskit.exceptions import MissingOptionalLibraryError, QiskitError
    from qiskit.qasm3 import QASM3ImporterError

    assert issubclass(QASM3ImporterError, QiskitError), (
        "QASM3ImporterError is no longer a QiskitError, so the executors may be "
        "catching the wrong exception type for their OpenQASM 2 fallback."
    )
    assert not issubclass(MissingOptionalLibraryError, QASM3ImporterError), (
        "A missing qasm3 importer would now be caught by the OpenQASM 2 fallback "
        "and silently mis-parse OpenQASM 3 circuits instead of failing loudly."
    )


def test_svm_kernel_sampler_defaults_to_an_aer_backend(svm_kernel: ModuleType):
    """Without a backend the svm kernel samples on a local Aer simulator.

    Qiskit's StatevectorSampler reference primitive was measured hundreds of
    times slower, so falling back to it again would be a performance regression.
    """
    from qiskit.primitives import BackendSamplerV2
    from qiskit_aer import AerSimulator

    sampler = svm_kernel._build_sampler(None, 0)

    assert isinstance(sampler, BackendSamplerV2), (
        f"Expected the fallback sampler to run on a real backend but got "
        f"{type(sampler).__name__}, which simulates the statevector in python."
    )
    assert isinstance(sampler._backend, AerSimulator), (
        f"Expected the fallback sampler to be backed by AerSimulator but got "
        f"{type(sampler._backend).__name__}."
    )


def test_svm_kernel_sampler_honours_the_requested_shot_count(svm_kernel: ModuleType):
    """A caller supplied shot count overrides the default."""
    sampler = svm_kernel._build_sampler(None, 512)

    assert sampler.options.default_shots == 512, (
        f"The requested shot count was not passed to the sampler, got "
        f"{sampler.options.default_shots} instead of 512."
    )


def test_svm_kernel_sampler_falls_back_to_the_default_shot_count(
    svm_kernel: ModuleType,
):
    """A missing shot count falls back to the documented default."""
    sampler = svm_kernel._build_sampler(None, 0)

    assert sampler.options.default_shots == svm_kernel.DEFAULT_SHOTS, (
        f"A shot count of 0 should fall back to DEFAULT_SHOTS "
        f"({svm_kernel.DEFAULT_SHOTS}) but the sampler uses "
        f"{sampler.options.default_shots}."
    )


def test_kernel_estimation_evaluates_a_kernel_matrix(
    kernel_estimation_kernel: ModuleType,
):
    """The kernel estimation plugin can compute a kernel matrix.

    This fails when the feature map is not decomposed before sampling, because
    BackendSamplerV2 rejects the composite feature map instruction.
    """
    import numpy as np

    kernel = kernel_estimation_kernel.KernelEnum.z_feature_map.get_kernel(
        None, 3, ["Z"], 1, "linear", 128
    )
    data = np.zeros((2, 3))

    matrix = kernel.evaluate(data)

    assert matrix.shape == (
        2,
        2,
    ), f"Expected a 2x2 kernel matrix for two data points but got {matrix.shape}."
    assert np.allclose(np.diag(matrix), 1.0), (
        f"A fidelity kernel must report a self similarity of 1 on its diagonal, "
        f"got {np.diag(matrix)}. The circuits were probably not executed correctly."
    )


def test_pennylane_aer_backend_runs_without_compatibility_patches():
    """The pennylane plugins reach a local Aer device on the pinned versions.

    The monkeypatches that faked an older qiskit version for pennylane-qiskit
    were removed, so this guards that they are really not needed any more.
    """
    pytest.importorskip("pennylane")
    pytest.importorskip("pennylane_qiskit")
    import pennylane as qml

    backends = load_plugin_module(
        "qcnn_quantum_backends",
        PENNYLANE_ML_PATH / "qcnn" / "backend" / "quantum_backends.py",
    )
    device = backends.QuantumBackends.aer_qasm_simulator.get_pennylane_backend(
        "", "", 2, 100
    )

    @qml.qnode(device)
    def circuit():
        qml.PauliX(wires=0)
        return qml.expval(qml.PauliZ(0))

    assert float(circuit()) == pytest.approx(-1.0), (
        "An X gate on |0> must measure -1 for PauliZ. The pennylane qiskit "
        "device did not execute the circuit as expected."
    )


def test_pennylane_ibmq_backend_reports_that_it_is_unavailable():
    """Selecting an IBMQ backend fails with a clear error.

    These backends cannot work on qiskit 2 because qiskit-ibm-provider is not
    compatible with it. Pinning the behaviour keeps the limitation visible.
    """
    pytest.importorskip("pennylane")
    pytest.importorskip("pennylane_qiskit")

    backends = load_plugin_module(
        "qcnn_quantum_backends_ibmq",
        PENNYLANE_ML_PATH / "qcnn" / "backend" / "quantum_backends.py",
    )

    with pytest.raises((RuntimeError, ValueError)):
        backends.QuantumBackends.ibmq_qasm_simulator.get_pennylane_backend(
            "token", "", 2, 100
        )


@pytest.fixture(scope="module")
def qiskit_executor_backends() -> ModuleType:
    """Return the backend module of the qiskit executor plugin."""
    return load_plugin_module(
        "qiskit_executor_backends",
        QISKIT_EXECUTOR_PATH / "backend" / "qiskit_backends.py",
    )


def test_runtime_channel_is_read_from_the_qiskit_ibm_variable(
    qiskit_executor_backends: ModuleType, monkeypatch
):
    """The current QISKIT_IBM_CHANNEL variable sets the runtime channel."""
    monkeypatch.setenv("QISKIT_IBM_CHANNEL", "ibm_cloud")
    monkeypatch.delenv("IBMQ_CHANNEL", raising=False)

    channel, _ = qiskit_executor_backends._runtime_config()

    assert channel == "ibm_cloud", (
        f"QISKIT_IBM_CHANNEL was ignored, the runtime would connect to '{channel}' "
        f"instead of the configured channel."
    )


def test_runtime_channel_falls_back_to_the_deprecated_variable(
    qiskit_executor_backends: ModuleType, monkeypatch
):
    """The deprecated IBMQ_CHANNEL variable still works as a fallback."""
    monkeypatch.delenv("QISKIT_IBM_CHANNEL", raising=False)
    monkeypatch.setenv("IBMQ_CHANNEL", "ibm_cloud")

    channel, _ = qiskit_executor_backends._runtime_config()

    assert channel == "ibm_cloud", (
        f"The deprecated IBMQ_CHANNEL fallback stopped working, existing "
        f"deployments configured with it would connect to '{channel}'."
    )


def test_runtime_channel_prefers_the_current_variable(
    qiskit_executor_backends: ModuleType, monkeypatch
):
    """The current variable wins when both spellings are set."""
    monkeypatch.setenv("QISKIT_IBM_CHANNEL", "ibm_cloud")
    monkeypatch.setenv("IBMQ_CHANNEL", "ibm_quantum_platform")

    channel, _ = qiskit_executor_backends._runtime_config()

    assert channel == "ibm_cloud", (
        f"The deprecated IBMQ_CHANNEL overrode the current QISKIT_IBM_CHANNEL, "
        f"the runtime would connect to '{channel}'."
    )


def test_runtime_channel_uses_the_default_when_unconfigured(
    qiskit_executor_backends: ModuleType, monkeypatch
):
    """An unset channel falls back to the documented default."""
    monkeypatch.delenv("QISKIT_IBM_CHANNEL", raising=False)
    monkeypatch.delenv("IBMQ_CHANNEL", raising=False)

    channel, _ = qiskit_executor_backends._runtime_config()

    assert channel == qiskit_executor_backends.DEFAULT_IBM_CHANNEL, (
        f"An unconfigured channel must default to DEFAULT_IBM_CHANNEL "
        f"({qiskit_executor_backends.DEFAULT_IBM_CHANNEL}) but got '{channel}'."
    )


def test_runtime_instance_is_none_when_unconfigured(
    qiskit_executor_backends: ModuleType, monkeypatch
):
    """An unset instance stays None so the runtime picks its own default."""
    monkeypatch.delenv("QISKIT_IBM_INSTANCE", raising=False)
    monkeypatch.delenv("IBMQ_INSTANCE", raising=False)

    _, instance = qiskit_executor_backends._runtime_config()

    assert instance is None, (
        f"An unconfigured instance must stay None, got '{instance}'. A non None "
        f"value would be passed to the runtime service as a real instance name."
    )


def test_runtime_channel_treats_an_empty_variable_as_unset(
    qiskit_executor_backends: ModuleType, monkeypatch
):
    """A variable left blank in a .env file must not become the channel name."""
    monkeypatch.setenv("QISKIT_IBM_CHANNEL", "")
    monkeypatch.delenv("IBMQ_CHANNEL", raising=False)

    channel, _ = qiskit_executor_backends._runtime_config()

    assert channel == qiskit_executor_backends.DEFAULT_IBM_CHANNEL, (
        f"An empty QISKIT_IBM_CHANNEL was taken literally, the runtime would be "
        f"asked to connect to the channel '{channel}'."
    )


def test_runtime_instance_treats_an_empty_variable_as_unset(
    qiskit_executor_backends: ModuleType, monkeypatch
):
    """A blank instance must stay None instead of an empty instance name."""
    monkeypatch.setenv("QISKIT_IBM_INSTANCE", "")
    monkeypatch.delenv("IBMQ_INSTANCE", raising=False)

    _, instance = qiskit_executor_backends._runtime_config()

    assert instance is None, (
        f"An empty QISKIT_IBM_INSTANCE was taken literally, the runtime would be "
        f"asked for the instance '{instance}' instead of picking its default."
    )


@pytest.fixture(scope="module")
def qiskit_simulator() -> ModuleType:
    """Return the qiskit simulator plugin module via the app factory.

    The module registers a celery task against the plugin instance, so it can
    only be imported once the plugin has been loaded by the app.
    """
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_qasm3_import")

    from logging import INFO
    import sys

    from qhana_plugin_runner import create_app

    create_app(
        {
            "SECRET_KEY": "test",
            "TESTING": True,
            "OPENAPI_VERSION": "3.0.2",
            "OPENAPI_JSON_PATH": "api-spec.json",
            "OPENAPI_URL_PREFIX": "",
            "DEFAULT_LOG_SEVERITY": INFO,
            "DEFAULT_FILE_STORE": "local_filesystem",
            "FILE_STORE_ROOT_PATH": "files",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "PLUGIN_FOLDERS": ["./stable_plugins/quantum_ml/qiskit_ml"],
            "DISABLED_PLUGINS": [],
        },
        silent_log=True,
    )
    module = next(
        (m for name, m in sys.modules.items() if name.endswith("qiskit_simulator")), None
    )
    assert module is not None, (
        "The qiskit simulator plugin was not loaded by the app factory, so the "
        "simulation code under test could not be reached."
    )
    return module


def test_simulator_counts_a_bell_state(qiskit_simulator: ModuleType):
    """Simulating a Bell state yields only the correlated outcomes."""
    _, counts, _ = qiskit_simulator.simulate_circuit(
        QASM3_CIRCUIT, {"shots": 256, "statevector": False}
    )

    assert sum(counts.values()) == 256, (
        f"The requested shot count was not simulated, the counts add up to "
        f"{sum(counts.values())} instead of 256."
    )
    unexpected = {
        k: v for k, v in counts.items() if k.replace(" ", "") not in ("00", "11")
    }
    assert not unexpected, (
        f"A Bell state can only collapse to 00 or 11, but the simulator also "
        f"reported {unexpected}."
    )


def test_simulator_returns_a_statevector_when_requested(qiskit_simulator: ModuleType):
    """Requesting a statevector actually produces one.

    AerSimulator only reports a statevector when the circuit carries a save
    instruction. Without it the result raises and the plugin silently returns
    None while still advertising a statevector output.
    """
    _, _, state_vector = qiskit_simulator.simulate_circuit(
        QASM3_CIRCUIT, {"shots": 64, "statevector": True}
    )

    assert state_vector is not None, (
        "No statevector was returned even though one was requested. The "
        "save_statevector instruction is probably missing from the circuit."
    )
    assert (
        len(state_vector) == 4
    ), f"A two qubit circuit must yield 4 amplitudes, got {len(state_vector)}."


def test_simulator_omits_the_statevector_when_not_requested(
    qiskit_simulator: ModuleType,
):
    """The statevector is only simulated when it was asked for."""
    _, _, state_vector = qiskit_simulator.simulate_circuit(
        QASM3_CIRCUIT, {"shots": 64, "statevector": False}
    )

    assert state_vector is None, (
        "A statevector was simulated even though it was not requested, which "
        "wastes a second simulation run of every circuit."
    )


def test_simulator_reports_the_backend_in_its_trace_metadata(
    qiskit_simulator: ModuleType,
):
    """The trace metadata names the backend that ran the circuit."""
    metadata, _, _ = qiskit_simulator.simulate_circuit(
        QASM3_CIRCUIT, {"shots": 64, "statevector": False}
    )

    assert metadata["qpuName"], (
        f"The trace metadata carries no qpuName ({metadata['qpuName']!r}), so the "
        f"backend name fallback stopped resolving."
    )
    assert metadata["shots"] == 64, (
        f"The trace metadata reports {metadata['shots']} shots instead of the 64 "
        f"that were simulated."
    )
