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

from qiskit_machine_learning.algorithms.classifiers import VQC
import numpy as np
from qiskit.utils import QuantumInstance
from enum import Enum
from qiskit.circuit.library import (
    TwoLocal,
    RealAmplitudes,
    ExcitationPreserving,
    EfficientSU2,
)
from .optimizer import OptimizerEnum


class AnsatzEnum(Enum):
    real_amplitudes = "RealAmplitudes"
    excitation_preserving = "ExcitationPreserving"
    efficient_su2 = "EfficientSU2"
    ry_rz = "RyRz"

    def get_ansatz(self, n_qbits: int, entanglement, reps: int):
        if self == AnsatzEnum.real_amplitudes:
            return RealAmplitudes(
                num_qubits=n_qbits, entanglement=entanglement, reps=reps
            )
        elif self == AnsatzEnum.excitation_preserving:
            return ExcitationPreserving(
                num_qubits=n_qbits, entanglement=entanglement, reps=reps
            )
        elif self == AnsatzEnum.efficient_su2:
            return EfficientSU2(num_qubits=n_qbits, entanglement=entanglement, reps=reps)
        elif self == AnsatzEnum.ry_rz:
            return TwoLocal(
                num_qubits=n_qbits,
                rotation_blocks=["ry", "rz"],
                entanglement_blocks="cz",
                entanglement=entanglement,
                reps=reps,
            )
        else:
            err_str = f"Unkown ansatz {self.name}!"
            raise ValueError(err_str)


class qiskitVQC:
    def __init__(
        self,
        quantum_instance: QuantumInstance,
        feature_map,
        ansatz,
        optimizer,
    ):
        self.__quantum_instance = quantum_instance
        self.__feature_map = feature_map
        self.__ansatz = ansatz
        self.__optimizer = optimizer
        self.__vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            quantum_instance=quantum_instance,
        )

    def prep_labels(self, labels):
        n_samples = len(labels)
        unique_labels = list(set(labels))
        idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        n_classes = len(list(set(labels)))

        # convert labels to one-hot
        labels_onehot = np.zeros((n_samples, n_classes))
        for i, ent_label in enumerate(labels):
            # We need a one hot encoding for labels
            labels_onehot[i, label_to_idx[ent_label]] = 1

        return labels_onehot, idx_to_label

    def fit(self, train_data: np.ndarray, labels: np.ndarray):
        # Prepare labels
        labels_onehot, self.__idx_to_label = self.prep_labels(labels)

        # fit vqc
        self.__vqc.fit(train_data, labels_onehot)

    def predict(self, test_data: np.ndarray):
        result = np.array(self.__vqc.predict(test_data))
        # convert back from one-hot to class
        label_indices = result.argmax(axis=1)
        labels = [self.__idx_to_label[idx] for idx in label_indices]
        return labels

    def get_representative_circuit(self, data, labels):
        # Init vqc
        if self.__vqc._fit_result is not None:
            vqc = self.__vqc
        else:
            optimizer = OptimizerEnum.cobyla.get_optimizer(1)
            vqc = VQC(
                feature_map=self.__feature_map,
                ansatz=self.__ansatz,
                optimizer=optimizer,
                quantum_instance=self.__quantum_instance,
            )
            labels_onehot, _ = self.prep_labels(labels)
            vqc.fit(data, labels_onehot)

        # Prep parameters
        param_values = {
            input_param: data[0, j]
            for j, input_param in enumerate(vqc._neural_network._input_params)
        }
        param_values.update(
            {
                weight_param: vqc._fit_result.x[j]
                for j, weight_param in enumerate(vqc._neural_network._weight_params)
            }
        )

        # Bind parameters to circuit
        circuit = vqc._neural_network._circuit.bind_parameters(param_values)

        # Return qasm string
        return circuit.qasm()

    def get_weights(self):
        if self.__vqc._fit_result is None:
            return None
        return self.__vqc._fit_result.x
