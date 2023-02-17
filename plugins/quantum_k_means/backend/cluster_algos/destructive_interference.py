# Copyright 2022 QHAna plugin runner contributors.
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

from .clustering import Clustering
import pennylane as qml
import numpy as np
from typing import List, Tuple


class DestructiveInterferenceQuantumKMeans(Clustering):
    def __init__(self, backend: qml.Device, tol, max_runs):
        super(DestructiveInterferenceQuantumKMeans, self).__init__(backend, tol, max_runs)
        # Number of qbits needed to calculate "distance" between one data point and one centroid
        self.needed_qbits = 2

    def init_centroids(self, k: int) -> np.ndarray:
        """
        Gets k random 2D points, then standardize and normalize them.
        """
        return self.normalize(np.random.uniform(size=(k, 2), low=-1, high=1))

    def prep_centroids(self, centroids) -> np.ndarray:
        """
        Normalizes the centroids.
        """
        return self.normalize(centroids)

    def calculate_angles(self, cartesian_points):
        """
        Calculates the angle between the 2D vetors and the base vector.
        The cartesian points are given in a tuple format (x, y).
        """
        base_vector = np.array([1, 0])
        angles = []
        for d_point in cartesian_points:
            if d_point[1] < 0:
                angles.append(2 * np.pi - np.arccos(np.dot(d_point, base_vector)))
            else:
                angles.append(np.arccos(np.dot(d_point, base_vector)))
        return np.array(angles)

    def prep_data(self, data) -> np.ndarray:
        """
        Prepares data, by standardizing and normalizing it.
        """
        return self.normalize(self.standardize(data))

    def prep_data_for_circuit(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare data for the quantum circuit by calculating the necessary angles for the algorithm.
        """
        return self.calculate_angles(data)

    def quantum_circuit(
        self, wires_to_use: List[int], data_angle: float, centroid_angle: float
    ):
        measure_wire = wires_to_use[0]
        ancilla_wire = wires_to_use[1]
        relative_angle = np.abs(data_angle - centroid_angle)
        qml.Hadamard(measure_wire)
        qml.CNOT(wires=[measure_wire, ancilla_wire])
        qml.RY(-relative_angle, wires=[ancilla_wire])
        qml.CNOT(wires=[measure_wire, ancilla_wire])
        qml.RY(relative_angle, wires=ancilla_wire)
        qml.Hadamard(measure_wire)

    def execute_circuit(
        self,
        current_distances_calculated,
        data_angles: List[float],
        centroid_angles: np.ndarray,
    ) -> Tuple[List[float], str]:
        @qml.qnode(self.backend)
        def circuit():
            wires_to_measure = []
            for wires_to_use, indices in current_distances_calculated:
                data_idx, centroid_idx = indices
                self.quantum_circuit(
                    wires_to_use, data_angles[data_idx], centroid_angles[centroid_idx]
                )
                wires_to_measure.append(wires_to_use[0])
            return [qml.probs(wires=[wire]) for wire in wires_to_measure]

        result = circuit()
        # Probability of measuring |1> and open qasm
        return [probs[1] for probs in result], circuit.tape.to_openqasm()

    def compute_new_centroid_mapping(
        self, prepped_data: List[float], centroids: List[List[float]]
    ) -> Tuple[np.ndarray, int, str]:
        # Need to return one of the quantum circuits
        representative_circuit = ""

        centroid_angles = self.calculate_angles(centroids)
        centroid_mapping = np.zeros(len(prepped_data), dtype=int)
        # Since we want the minimum result of our quantum circuits and the results lie within [0, 1],
        # we can start out with 1.
        mapping_distance = np.ones(len(prepped_data))

        next_qbit = 0  # this tracks the next free qbit
        amount_executed_circuits = 0
        current_distances_calculated = []
        for i in range(len(prepped_data)):
            for j in range(len(centroid_angles)):
                # If if-statement is true, then we don't have enough qbits left to prepare another circuit
                # Therefore execute the circuits on a quantum computer
                if next_qbit + self.needed_qbits > self.max_qbits:
                    amount_executed_circuits += 1
                    # This adds the measurements and executes the circuits
                    results, representative_circuit = self.execute_circuit(
                        current_distances_calculated, prepped_data, centroid_angles
                    )
                    # Update centroid mapping
                    for k in range(len(results)):
                        data_idx, centroid_idx = current_distances_calculated[k][1]
                        # If result = 0, then the centroid and the data point are close to each other
                        # If result = 1, then the centroid and the data point are far away from each other
                        # Therefore we look at which is min
                        if mapping_distance[data_idx] > results[k]:
                            mapping_distance[data_idx] = results[k]
                            centroid_mapping[data_idx] = centroid_idx
                    # Reset variables
                    next_qbit = 0
                    current_distances_calculated = []

                # Prepare quantum circuit
                wires_to_use = [i + next_qbit for i in range(self.needed_qbits)]
                current_distances_calculated.append([wires_to_use, [i, j]])
                next_qbit += self.needed_qbits

        if next_qbit != 0:
            amount_executed_circuits += 1
            # This adds the measurements and executes the circuits
            results, representative_circuit = self.execute_circuit(
                current_distances_calculated, prepped_data, centroid_angles
            )
            # Update centroid mapping
            for k in range(len(results)):
                data_idx, centroid_idx = current_distances_calculated[k][1]
                # We want min
                if mapping_distance[data_idx] > results[k]:
                    mapping_distance[data_idx] = results[k]
                    centroid_mapping[data_idx] = centroid_idx

        return centroid_mapping, amount_executed_circuits, representative_circuit

    def plot(self, prepped_data, prepped_centroids, centroid_mapping):
        import plotly.express as px
        import pandas as pd

        points_x = []
        points_y = []
        colors = []
        ids = []

        for i in range(len(prepped_data)):
            points_x.append(prepped_data[i][0])
            points_y.append(prepped_data[i][1])
            ids.append(str(i))

            if centroid_mapping[i] == 0:
                colors.append("red")
            elif centroid_mapping[i] == 1:
                colors.append("blue")
            else:
                raise ValueError("Too many clusters.")

        for c in range(len(prepped_centroids)):
            points_x.append(prepped_centroids[c][0])
            points_y.append(prepped_centroids[c][1])
            colors.append("green")
            ids.append(f"c{c}")

        df = pd.DataFrame(
            {
                "x": points_x,
                "y": points_y,
                "ID": ids,
                "cluster": colors,
                "size": [10 for _ in range(len(points_x))],
            }
        )

        fig = px.scatter(
            df,
            x="x",
            y="y",
            hover_name="ID",
            color="cluster",
            symbol="cluster",
            size="size",
        )
        fig.show()
