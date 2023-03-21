import enum

from celery.utils.log import get_task_logger
from qiskit import IBMQ, Aer
from qiskit.primitives import BackendSampler


TASK_LOGGER = get_task_logger(__name__)


class QiskitBackendEnum(enum.Enum):
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
    custom_ibmq = "custom_ibmq"

    def get_qiskit_backend(
        self,
        ibmq_token: str,
        custom_backend_name: str,
        shots: int,
    ) -> BackendSampler:
        if self.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = self.name[4:]

            backend = Aer.get_backend(aer_backend_name)
        elif self.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            backend = provider.get_backend(self.name)
        elif self.name.startswith("custom_ibmq"):
            # Use custom IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            backend = provider.get_backend(custom_backend_name)
        else:
            raise NotImplementedError("Unknown qiskit backend specified!")

        sampler = BackendSampler(backend, options=dict(shots=shots))
        return sampler
