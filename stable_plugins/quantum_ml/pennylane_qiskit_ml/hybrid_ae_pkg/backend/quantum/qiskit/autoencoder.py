# Copyright 2023 QHAna plugin runner contributors.
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

from typing import List, Tuple, Dict

import numpy as np
import torch
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    execute,
    ClassicalRegister,
    BasicAer,
    transpile,
    assemble,
)
from qiskit.circuit import Gate, ControlledGate, Parameter, Instruction
from qiskit.providers.aer import QasmSimulator


# TODO: measure performance
class QuantumAutoencoder:
    """
    Implements the Quantum Autoencoder described in
    J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.
    """

    def __init__(self, input_dim: int, embedding_dim: int, unit_cell_cnt: int):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.unit_cell_cnt = unit_cell_cnt

        input_qr = QuantumRegister(
            self.input_dim
        )  # quantum register for the compressed state and the trash state
        # quantum register for the second quantum autoencoder that is used for calculating the fidelity between
        # two quantum embeddings
        input2_qr = QuantumRegister(self.input_dim)
        ref_qr = QuantumRegister(
            self.input_dim - self.embedding_dim
        )  # quantum register for the reference state
        swap_out_qr = QuantumRegister(
            1
        )  # quantum register for the output of the SWAP test
        swap_out_cr = ClassicalRegister(
            1
        )  # classical register for the output of the SWAP test
        embedding_cr = ClassicalRegister(
            self.embedding_dim
        )  # classical register for the embedding
        reconstruction_cr = ClassicalRegister(
            self.input_dim
        )  # classical register for the output of the decoder

        self.input_encoder_circ = QuantumCircuit(
            input_qr
        )  # quantum circuit with input layer and encoder

        # input
        input_layer, self._input_params = self._create_input_inst(self.input_dim)
        self.input_encoder_circ.append(input_layer, input_qr)
        self.input_encoder_circ.barrier()

        # encoder
        encoder, self._encoder_params = self._create_encoder_inst(
            self.input_dim, self.unit_cell_cnt
        )
        self.input_encoder_circ.append(encoder, input_qr)
        self.input_encoder_circ.barrier()

        # construct the circuit for training
        # calculate fidelity between the trash state and the reference state via a SWAP test
        self.train_circ = self.input_encoder_circ.copy()
        self.train_circ.add_register(ref_qr, swap_out_qr, swap_out_cr)
        swap_test = self._create_swap_test_inst(self.input_dim - self.embedding_dim)
        self.train_circ.append(
            swap_test,
            [swap_out_qr[0]]
            + ref_qr[:]
            + input_qr[0 : (self.input_dim - self.embedding_dim)],
        )

        # measure SWAP test result
        self.train_circ.measure(swap_out_qr[0], swap_out_cr[0])

        self.backend = QasmSimulator()
        self.transpiled_train_circ = transpile(self.train_circ, backend=self.backend)

        # construct the circuit for measuring the embeddings
        self.embedding_circ = self.input_encoder_circ.copy()
        self.embedding_circ.add_register(embedding_cr)

        for i in range(self.input_dim - self.embedding_dim):
            self.embedding_circ.reset(input_qr[i])

        self.embedding_circ.measure(
            input_qr[(self.input_dim - self.embedding_dim) :], embedding_cr
        )

        # construct the circuit for the quantum embedding
        self.quantum_embedding_circ = self.input_encoder_circ.copy()
        self.quantum_embedding_circ.add_register(embedding_cr)

        for i in range(self.input_dim - self.embedding_dim):
            self.quantum_embedding_circ.reset(input_qr[i])

        # construct the circuit for measuring the output of the decoder
        self.reconstruction_circ = self.input_encoder_circ.copy()
        self.reconstruction_circ.add_register(reconstruction_cr)

        for i in range(self.input_dim - self.embedding_dim):
            self.reconstruction_circ.reset(input_qr[i])

        self.reconstruction_circ.barrier()

        decoder = encoder.inverse()
        self.reconstruction_circ.append(decoder, input_qr)
        self.reconstruction_circ.measure(input_qr, reconstruction_cr)

        self.transpiled_recon_circ = transpile(
            self.reconstruction_circ, backend=self.backend
        )

        # construct the circuit for measuring the fidelity between two quantum embeddings
        self.fidelity_circ = QuantumCircuit(input_qr, input2_qr, swap_out_qr, swap_out_cr)

        input_layer1, self._input_params1 = self._create_input_inst(
            self.input_dim, prefix="Input1"
        )
        input_layer2, self._input_params2 = self._create_input_inst(
            self.input_dim, prefix="Input2"
        )
        self.fidelity_circ.append(input_layer1, input_qr)
        self.fidelity_circ.append(input_layer2, input2_qr)
        self.input_encoder_circ.barrier()

        encoder1, self._encoder_params1 = self._create_encoder_inst(
            self.input_dim, self.unit_cell_cnt, "1_"
        )
        encoder2, self._encoder_params2 = self._create_encoder_inst(
            self.input_dim, self.unit_cell_cnt, "2_"
        )
        self.fidelity_circ.append(encoder1, input_qr)
        self.fidelity_circ.append(encoder2, input2_qr)
        self.fidelity_circ.barrier()

        swap_test = self._create_swap_test_inst(input_dim)
        self.fidelity_circ.append(swap_test, [swap_out_qr] + input_qr[:] + input2_qr[:])
        self.fidelity_circ.measure(swap_out_qr, swap_out_cr)

        self.transpiled_fidelity_circ = transpile(
            self.fidelity_circ, backend=self.backend
        )

    def run(self, input_values: torch.Tensor, encoder_weights: torch.Tensor, shots: int):
        """
        :param input_values: size: [instance_cnt, input_cnt]
        :param encoder_weights: size: [instance_cnt, encoder_params_cnt]
        :param decoder_params: size: [instance_cnt, decoder_params_cnt]
        :param shots: number of shots
        :return:
        """
        parameter_binds = self._create_param_binds(
            input_values, self._input_params, encoder_weights, self._encoder_params
        )
        qobj = assemble(
            [
                self.transpiled_train_circ.bind_parameters(value_dict)
                for value_dict in parameter_binds
            ]
        )

        job = self.backend.run(qobj)
        counts: List[Dict[str, int]] = job.result().get_counts()

        if isinstance(counts, Dict):
            counts = [counts]

        zero_prob = 1 - self._extract_single_qubit_expectations(
            counts, 1, shots
        )  # probability that 0 is measured

        return zero_prob

    def calc_embedding(
        self, input_values: torch.Tensor, encoder_weights: torch.Tensor, shots: int
    ):
        parameter_binds = self._create_param_binds(
            input_values, self._input_params, encoder_weights, self._encoder_params
        )

        job = execute(
            self.embedding_circ,
            backend=QasmSimulator(),
            shots=shots,
            parameter_binds=parameter_binds,
        )
        counts: List[Dict[str, int]] = job.result().get_counts()

        if isinstance(counts, Dict):
            counts = [counts]

        embeddings = self._extract_single_qubit_expectations(
            counts, self.embedding_dim, shots
        )

        return embeddings

    def calc_fidelity(
        self,
        input1: np.ndarray,
        input2: np.ndarray,
        encoder_weights: np.ndarray,
        shots: int,
    ) -> float:
        bind = {}

        for i in range(input1.shape[0]):
            bind[self._input_params1[i]] = input1[i]
            bind[self._input_params2[i]] = input2[i]

        for i in range(encoder_weights.shape[0]):
            bind[self._encoder_params1[i]] = encoder_weights[i].item()
            bind[self._encoder_params2[i]] = encoder_weights[i].item()

        qobj = assemble(
            [
                self.transpiled_fidelity_circ.bind_parameters(value_dict)
                for value_dict in [bind]
            ]
        )
        job = self.backend.run(qobj)

        counts = job.result().get_counts()
        expectation: float = self._extract_single_qubit_expectations(
            [counts], 1, shots
        ).numpy()[0, 0]
        fidelity = max(1 - (2.0 * expectation), 0)

        return fidelity

    def calc_embedding_statevector(
        self, input_values: torch.Tensor, encoder_weights: torch.Tensor, shots: int
    ):
        parameter_binds = self._create_param_binds(
            input_values, self._input_params, encoder_weights, self._encoder_params
        )

        job = execute(
            self.quantum_embedding_circ,
            backend=BasicAer.get_backend("statevector_simulator"),
            shots=shots,
            parameter_binds=parameter_binds,
        )
        result = job.result()

        statevectors = [result.get_statevector(i) for i in range(len(result.results))]
        embedding_states = []

        sv: np.ndarray

        for sv in statevectors:
            # remove reset qubits
            embedding_states.append(
                sv.reshape((-1, 2 ** (self.input_dim - self.embedding_dim)))[:, 0]
            )

        return embedding_states

    def calc_reconstructions(
        self, input_values: torch.Tensor, encoder_weights: torch.Tensor, shots: int
    ):
        parameter_binds = self._create_param_binds(
            input_values, self._input_params, encoder_weights, self._encoder_params
        )

        qobj = assemble(
            [
                self.transpiled_recon_circ.bind_parameters(value_dict)
                for value_dict in parameter_binds
            ]
        )
        job = self.backend.run(qobj)
        counts: List[Dict[str, int]] = job.result().get_counts()

        if isinstance(counts, Dict):
            counts = [counts]

        reconstructions = self._extract_single_qubit_expectations(
            counts, self.input_dim, shots
        )

        return reconstructions

    @staticmethod
    def _create_rotation_gate(
        layer: int, qubit_id: int, param_prefix: str = ""
    ) -> Tuple[Gate, List[Parameter]]:
        id_str = param_prefix + str(layer) + "_" + str(qubit_id)
        params = [
            Parameter("Z1_" + id_str),
            Parameter("Y_" + id_str),
            Parameter("Z2_" + id_str),
        ]

        rotation_circuit = QuantumCircuit(1, name="R_" + id_str)
        rotation_circuit.rz(params[0], 0)
        rotation_circuit.ry(params[1], 0)
        rotation_circuit.rz(params[2], 0)
        rotation_gate: Gate = rotation_circuit.to_gate()

        return rotation_gate, params

    @staticmethod
    def _create_controlled_rotation_gate(
        layer: int, qubit_id: int, param_prefix: str = ""
    ) -> Tuple[ControlledGate, List[Parameter]]:
        gate, params = QuantumAutoencoder._create_rotation_gate(
            layer, qubit_id, param_prefix
        )

        return gate.control(), params

    @staticmethod
    def _add_input_layer(
        circuit: QuantumCircuit, input_qubits: QuantumRegister, prefix: str = "Input_"
    ) -> List[Parameter]:
        params = []

        for i, qubit in enumerate(input_qubits):
            param = Parameter(prefix + str(i))
            circuit.rx(param, qubit)
            params.append(param)

        return params

    @staticmethod
    def _add_rotation_layer(
        circuit: QuantumCircuit,
        register: QuantumRegister,
        layer: int,
        param_prefix: str = "",
    ) -> List[Parameter]:
        all_params = []

        for i, qubit in enumerate(register):
            gate, params = QuantumAutoencoder._create_rotation_gate(
                layer, i, param_prefix
            )
            circuit.append(gate, [qubit], [])
            all_params.extend(params)

        return all_params

    @staticmethod
    def _add_controlled_rotation_layer(
        circuit: QuantumCircuit,
        qr: QuantumRegister,
        start_layer: int,
        param_prefix: str = "",
    ) -> List[Parameter]:
        all_params = []

        for i in range(len(qr)):
            circuit.barrier()

            for j in range(len(qr)):
                if i != j:
                    gate, params = QuantumAutoencoder._create_controlled_rotation_gate(
                        start_layer + i, j, param_prefix
                    )
                    circuit.append(gate, [qr[i], qr[j]], [])
                    all_params.extend(params)

        return all_params

    @staticmethod
    def _create_input_inst(
        input_dim: int, prefix: str = "Input"
    ) -> Tuple[Instruction, List[Parameter]]:
        qr = QuantumRegister(input_dim)
        circ = QuantumCircuit(qr, name="Input")
        params = QuantumAutoencoder._add_input_layer(circ, qr, prefix)
        inst = circ.to_instruction()

        return inst, params

    @staticmethod
    def _create_encoder_inst(
        input_dim: int, unit_cell_cnt: int, param_prefix: str = ""
    ) -> Tuple[Instruction, List[Parameter]]:
        qr = QuantumRegister(input_dim)
        encoder = QuantumCircuit(qr, name="Encoder")
        params = []
        current_layer = 0

        params.extend(
            QuantumAutoencoder._add_rotation_layer(
                encoder, qr, current_layer, param_prefix
            )
        )
        current_layer += 1

        for _ in range(unit_cell_cnt):
            params.extend(
                QuantumAutoencoder._add_controlled_rotation_layer(
                    encoder, qr, current_layer, param_prefix
                )
            )
            current_layer += input_dim

            encoder.barrier()
            params.extend(
                QuantumAutoencoder._add_rotation_layer(
                    encoder, qr, current_layer, param_prefix
                )
            )
            current_layer += 1

        inst = encoder.to_instruction()

        return inst, params

    @staticmethod
    def _create_swap_test_inst(trash_state_dim: int) -> Instruction:
        out = QuantumRegister(1)
        reference_state = QuantumRegister(trash_state_dim)
        trash_state = QuantumRegister(trash_state_dim)
        circ = QuantumCircuit(out, reference_state, trash_state, name="SWAP test")

        circ.h(out[0])

        for i in range(trash_state_dim):
            circ.cswap(out[0], reference_state[i], trash_state[i])

        circ.h(out[0])

        return circ.to_instruction()

    @staticmethod
    def _create_param_binds(
        input_values: torch.Tensor,
        input_params: List[Parameter],
        encoder_weights: torch.Tensor,
        encoder_params: List[Parameter],
    ) -> List[Dict[Parameter, float]]:
        parameter_binds = []

        for i in range(input_values.shape[0]):
            param_bind = {}

            for j, input_param in enumerate(input_params):
                param_bind[input_param] = input_values[i, j].item()

            for j, encoder_param in enumerate(encoder_params):
                param_bind[encoder_param] = encoder_weights[i, j].item()

            parameter_binds.append(param_bind)

        return parameter_binds

    @staticmethod
    def _extract_single_qubit_expectations(
        counts: List[Dict[str, int]], num_clbits: int, shots: int
    ):
        expectations = torch.zeros((len(counts), num_clbits))

        for i, experiment_results in enumerate(counts):
            for k, v in experiment_results.items():
                for j in range(num_clbits):
                    if (
                        k[num_clbits - j - 1] == "1"
                    ):  # the string is little-endian (cr[0] is on the right hand side)
                        expectations[i, j] += v

        expectations /= shots

        return expectations

    @staticmethod
    def draw_to_file(
        circ: QuantumCircuit,
        filename: str,
        decompose_levels: int = 0,
        output_format: str = "mpl",
    ):
        circuit = circ

        for _ in range(decompose_levels):
            circuit = circuit.decompose()

        circuit.draw(output=output_format, filename=filename)

    def get_encoder_param_cnt(self) -> int:
        return len(self._encoder_params)
