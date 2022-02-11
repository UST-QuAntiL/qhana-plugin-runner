from threading import Lock
from typing import List

import mlflow
from pyquil import Program


class CircuitLogger:
    def __init__(self):
        self.lock = Lock()
        self.log_index = 0

    def log_circuit_execution(
        self,
        circuit: Program,
        input: List[float],
        parameters: List[float],
        response_time: float,
        output: List[List[float]],
    ):
        with self.lock:
            mlflow.log_dict(
                {
                    "circuit": circuit.out(),
                    "input": input,
                    "parameters": parameters,
                    "response_time": response_time,
                    "output": output,
                },
                "{:06}.json".format(self.log_index),
            )
            self.log_index += 1
