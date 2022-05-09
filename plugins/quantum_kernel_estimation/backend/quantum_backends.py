import enum

import pennylane as qml
from celery.utils.log import get_task_logger
from qiskit import IBMQ

TASK_LOGGER = get_task_logger(__name__)


class QuantumBackends(enum.Enum):
    custom_ibmq = "custom_ibmq"
    aer_statevector_simulator = "aer_statevector_simulator"
    aer_qasm_simulator = "aer_qasm_simulator"
    ibmq_qasm_simulator = "ibmq_qasm_simulator"
    ibmq_santiago = "ibmq_santiago"
    ibmq_manila = "ibmq_manila"
    ibmq_bogota = "ibmq_bogota"
    ibmq_quito = "ibmq_quito"
    ibmq_belem = "ibmq_belem"
    ibmq_lima = "ibmq_lima"
    ibmq_armonk = "ibmq_armonk"

    @staticmethod
    def get_pennylane_backend(
        backend_enum: "QuantumBackends",
        ibmq_token: str,
        custom_backend_name: str,
        qubit_cnt: int,
    ) -> qml.Device:
        if backend_enum.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = backend_enum.name[4:]

            return qml.device("qiskit.aer", wires=qubit_cnt, backend=aer_backend_name)
        elif backend_enum.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=backend_enum.name,
                provider=provider
            )
        elif backend_enum.name.startswith("custom_ibmq"):
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=custom_backend_name,
                provider=provider,
            )
        else:
            TASK_LOGGER.error("Unknown pennylane backend specified!")
