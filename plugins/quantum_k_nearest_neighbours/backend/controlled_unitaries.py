import pennylane as qml
import qiskit


def get_controlled_one_qubit_unitary(U):
    qc = qiskit.QuantumCircuit(1)
    qc.unitary(U, [0])
    c_qc = qc.control()
    sv_backend = qiskit.Aer.get_backend("statevector_simulator")
    qc_transpiled = qiskit.transpile(
        c_qc,
        backend=sv_backend,
        basis_gates=sv_backend.configuration().basis_gates,
        optimization_level=3,
    )
    converted = qml.from_qiskit(qc_transpiled)

    def circuit(c_wire, t_wire):
        converted(wires=(t_wire, c_wire))

    return circuit
