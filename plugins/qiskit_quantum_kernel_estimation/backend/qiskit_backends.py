import enum

from celery.utils.log import get_task_logger
from qiskit import IBMQ, Aer

TASK_LOGGER = get_task_logger(__name__)


class QiskitBackends(enum.Enum):
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
    def get_qiskit_backend(
        backend_enum: "QuantumBackends",
        ibmq_token: str,
        custom_backend_name: str,
    ):
        if backend_enum.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = backend_enum.name[4:]

            return Aer.get_backend(aer_backend_name)
        elif backend_enum.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return provider.get_backend(backend_enum.name)
        elif backend_enum.name.startswith("custom_ibmq"):
            # Use custom IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return provider.get_backend(custom_backend_name)
        else:
            TASK_LOGGER.error("Unknown qiskit backend specified!")
