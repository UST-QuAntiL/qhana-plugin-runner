import time
from multiprocessing.pool import ThreadPool
from typing import Any, Tuple, List, Callable, Optional

import mlflow
from pyquil import Program, get_qc
from pyquil.api import QuantumComputer
import torch
from torch import Tensor
from torch.autograd import Function
import numpy as np
from scipy.optimize import minimize

from plugins.hybrid_ae_pkg.backend.gradient_free import (
    scipy_compatible_objective_function,
    get_number_of_parameters,
    set_parameters_from_1D_array,
)
from plugins.hybrid_ae_pkg.backend.quantum.pytorch_backend import pyquil_backend
from plugins.hybrid_ae_pkg.backend.quantum.pytorch_backend.pyquil_backend import (
    QNN1,
    QNN2,
    QNN3,
    QNN4,
)
from plugins.hybrid_ae_pkg.backend.quantum.pytorch_backend.pyquil_backend.circuit_logging import (
    CircuitLogger,
)


class PyquilFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        program: Program,
        qc: QuantumComputer,
        input_data: Tensor,
        input_region_name: str,
        params: Tensor,
        params_region_name: str,
        shift: float,
        logger: Optional[CircuitLogger] = None,
    ) -> Tensor:
        ctx.program = program
        ctx.qc = qc
        ctx.input_region_name = input_region_name
        ctx.params_region_name = params_region_name
        ctx.shift = shift
        ctx.logger = logger
        ctx.save_for_backward(input_data, params)

        params_extended = params.reshape((1, -1)).repeat(input_data.shape[0], 1)

        return PyquilFunction._calc_output(
            program,
            qc,
            input_data,
            input_region_name,
            params_extended,
            params_region_name,
            logger,
        )

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tuple[None, None, Tensor, None, Tensor, None, None, None]:
        input_data: Tensor
        params: Tensor
        input_data, params = ctx.saved_tensors
        program: Program = ctx.program
        qc: QuantumComputer = ctx.qc
        input_region_name: str = ctx.input_region_name
        params_region_name: str = ctx.params_region_name
        shift: float = ctx.shift
        logger: Optional[CircuitLogger] = ctx.logger

        # gradient approximation with central differences for the input data
        params_extended = params.reshape((1, -1)).repeat(2, 1)
        input_grad = torch.zeros(input_data.shape)

        for i in range(input_data.shape[0]):
            for j in range(input_data.shape[1]):
                right_shift = input_data[i].clone().detach()
                right_shift[j] += shift
                left_shift = input_data[i].clone().detach()
                left_shift[j] -= shift

                shifted_input = torch.stack([right_shift, left_shift], dim=0)
                output = PyquilFunction._calc_output(
                    program,
                    qc,
                    shifted_input,
                    input_region_name,
                    params_extended,
                    params_region_name,
                    logger,
                )

                derivatives = (output[0, :] - output[1, :]) / (2.0 * shift)
                combined_derivative = (derivatives * grad_output[i]).sum()

                input_grad[i, j] = combined_derivative

        # gradient approximation with central differences for the parameters
        params_grad = torch.zeros(params.shape)
        shifted_params = []

        for i in range(params.shape[0]):
            for j in range(input_data.shape[0]):
                right_shift = params.clone().detach()
                right_shift[i] += shift
                left_shift = params.clone().detach()
                left_shift[i] -= shift

                shifted_params.append(right_shift)
                shifted_params.append(left_shift)

        extended_input_values = input_data.repeat_interleave(2, 0).repeat(
            (params.shape[0], 1)
        )
        stacked_params = torch.stack(shifted_params, dim=0)
        output = PyquilFunction._calc_output(
            program,
            qc,
            extended_input_values,
            input_region_name,
            stacked_params,
            params_region_name,
            logger,
        )

        for i in range(params.shape[0]):
            for j in range(input_data.shape[0]):
                derivative = (
                    output[2 * (i * input_data.shape[0] + j)]
                    - output[2 * (i * input_data.shape[0] + j) + 1]
                ) / (2.0 * shift)
                combined_derivative = (derivative * grad_output[j]).sum()

                params_grad[i] += combined_derivative

        return None, None, input_grad, None, params_grad, None, None, None

    @staticmethod
    def _calc_output(
        program: Program,
        qc: QuantumComputer,
        input_data: Tensor,
        input_region_name: str,
        params: Tensor,
        params_region_name: str,
        logger: Optional[CircuitLogger] = None,
    ) -> Tensor:
        input_count = input_data.shape[0]
        programs = [program.copy() for _ in range(input_count)]
        inputs = [input_data[i].tolist() for i in range(input_count)]

        if params.dim() == 1:
            params = params.reshape((1, -1))

        params_split = [params[i].tolist() for i in range(input_count)]

        def run(
            program_instance: Program,
            single_input: List[float],
            single_parameters: List[float],
        ) -> np.ndarray:
            program_instance.write_memory(
                region_name=input_region_name, value=single_input
            )
            program_instance.write_memory(
                region_name=params_region_name, value=single_parameters
            )

            time1 = time.time()
            bit_strings = qc.run(program_instance).readout_data.get("ro")
            time2 = time.time()

            if logger is not None:
                logger.log_circuit_execution(
                    program_instance,
                    single_input,
                    single_parameters,
                    time2 - time1,
                    bit_strings.tolist(),
                )

            return np.mean(bit_strings, 0)

        with ThreadPool(8) as pool:
            output_arrays = pool.starmap(
                run,
                zip(programs, inputs, params_split),
            )

        probabilities = torch.tensor(np.stack(output_arrays), dtype=torch.float32)

        # convert from probabilities to expectation value of Z-measurement
        exp_values = 1.0 - (2.0 * probabilities)

        return exp_values


class PyQuilLayer(torch.nn.Module):
    def __init__(
        self,
        program: Program,
        qc: QuantumComputer,
        input_region_name: str,
        params_region_name: str,
        params_num: int,
        shift: float,
        logger: Optional[CircuitLogger] = None,
    ):
        super(PyQuilLayer, self).__init__()
        self.program = program
        self.qc = qc
        self.input_region_name = input_region_name
        self.params_region_name = params_region_name
        self.params_num = params_num
        self.shift = shift
        self.logger = logger

        self.params = torch.nn.Parameter(torch.rand(self.params_num) * 2 * np.pi)

    def forward(self, input_data: Tensor) -> Tensor:
        return PyquilFunction.apply(
            self.program,
            self.qc,
            input_data,
            self.input_region_name,
            self.params,
            self.params_region_name,
            self.shift,
            self.logger,
        )


def create_qlayer(
    constructor_func: Callable[[int, int], Tuple[Program, int]],
    q_num: int,
    layer_num: int,
    dev: QuantumComputer,
    logger: Optional[CircuitLogger] = None,
) -> PyQuilLayer:
    """
    Input of the created quantum layer should be in the range [0, pi]. The output will be in the range [-1, 1].

    @param constructor_func: Function that constructs the circuit.
    @param q_num: Number of qubits.
    @param layer_num: Number of layers.
    @param dev: device on which the circuits will be executed
    @param logger: logger that logs circuits, inputs, parameters and the result
    :return: PyTorch module with integrated PyQuil circuit.
    """
    program, param_num = constructor_func(q_num, layer_num)
    program.wrap_in_numshots_loop(1000)
    qlayer = PyQuilLayer(
        dev.compile(program), dev, "input", "params", param_num, 0.1, logger
    )

    return qlayer


qnn_constructors = {
    "QNN1": QNN1.create_circuit,
    "QNN2": QNN2.create_circuit,
    "QNN3": QNN3.create_circuit,
    "QNN4": QNN4.create_circuit,
}


def _gradient_based_optimization_example():
    program, params_num = QNN3.create_circuit(2, 2)
    program.wrap_in_numshots_loop(1000)
    qc = get_qc("4q-qvm")
    executable = qc.compile(program)

    training_input = torch.tensor(
        [[0.0, 0.0], [0.0, np.pi], [np.pi, 0.0], [np.pi, np.pi]]
    )
    training_target = torch.tensor([-1.0, 1.0, 1.0, -1.0])
    model = PyQuilLayer(executable, qc, "input", "params", params_num, 0.1)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    model.train()

    for i in range(100):
        time1 = time.time()
        pred: Tensor = model(training_input)
        loss = loss_fn(pred[:, 0], training_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time2 = time.time()

        print(f"loss: {loss.item():>7f} output: {pred} time: {time2 - time1}")


def gradient_free_optimization_experiment():
    qnn_name = "QNN3"
    q_num = 2
    layer_num = 1
    program, params_num = pyquil_backend.layer.qnn_constructors[qnn_name](
        q_num, layer_num
    )
    program.wrap_in_numshots_loop(1000)
    use_simulator = True
    qc = get_qc("Aspen-11", as_qvm=use_simulator)
    executable = qc.compile(program)

    training_input = torch.tensor(
        [[0.0, 0.0], [0.0, np.pi], [np.pi, 0.0], [np.pi, np.pi]]
    )
    training_target = torch.tensor([-1.0, 1.0, 1.0, -1.0])
    logger = CircuitLogger()
    model = PyQuilLayer(executable, qc, "input", "params", params_num, 0.1, logger)
    step = 0

    def loss_fn(output: torch.Tensor, target: torch.Tensor):
        nonlocal step
        loss = torch.nn.MSELoss()(output[:, 0], target)
        mlflow.log_metric("mse_loss", loss.item(), step)
        step += 1

        return loss

    initial_params = np.random.random(get_number_of_parameters(model)) * 2.0 * np.pi

    method = "COBYLA"
    # method = "Nelder-Mead"

    mlflow.log_param("QNN", qnn_name)
    mlflow.log_param("Number of qubits", q_num)
    mlflow.log_param("Number of layers", layer_num)
    mlflow.log_param("Use simulator", use_simulator)
    mlflow.log_param("Method", method)
    mlflow.log_param("QC", qc.name)
    time1 = time.time()

    res = minimize(
        scipy_compatible_objective_function,
        initial_params,
        (model, training_input, training_target, loss_fn),
        method,
        options={"disp": True},
    )

    time2 = time.time()

    set_parameters_from_1D_array(model, res.x)
    print(model(training_input))
    print(str(time2 - time1) + "s")


if __name__ == "__main__":
    gradient_free_optimization_experiment()
    # _gradient_based_optimization_example()
