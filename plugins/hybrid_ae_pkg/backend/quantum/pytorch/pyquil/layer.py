import time
from multiprocessing.pool import ThreadPool
from typing import Any, Tuple, List

from pyquil import Program, get_qc
from pyquil.api import QuantumComputer
import torch
from torch import Tensor
from torch.autograd import Function
import numpy as np

from plugins.hybrid_ae_pkg.backend.quantum.pytorch.pyquil import QNN1


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
    ) -> Tensor:
        ctx.program = program
        ctx.qc = qc
        ctx.input_region_name = input_region_name
        ctx.params_region_name = params_region_name
        ctx.shift = shift
        ctx.save_for_backward(input_data, params)

        params_extended = params.reshape((1, -1)).repeat(input_data.shape[0], 1)

        return PyquilFunction._calc_output(
            program,
            qc,
            input_data,
            input_region_name,
            params_extended,
            params_region_name,
        )

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tuple[None, None, Tensor, None, Tensor, None, None]:
        input_data: Tensor
        params: Tensor
        input_data, params = ctx.saved_tensors
        program: Program = ctx.program
        qc: QuantumComputer = ctx.qc
        input_region_name: str = ctx.input_region_name
        params_region_name: str = ctx.params_region_name
        shift: float = ctx.shift

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
                )

                derivatives = (output[0] - output[1]) / (2.0 * shift)
                combined_derivative = (derivatives * grad_output).sum()

                input_grad[i, j] = combined_derivative

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
        )

        for i in range(params.shape[0]):
            for j in range(input_data.shape[0]):
                derivatives = (
                    output[2 * (i * input_data.shape[0] + j)]
                    - output[2 * (i * input_data.shape[0] + j) + 1]
                ) / (2.0 * shift)
                combined_derivative = (derivatives * grad_output).sum()

                params_grad[i] += combined_derivative

        return None, None, input_grad, None, params_grad, None, None

    @staticmethod
    def _calc_output(
        program: Program,
        qc: QuantumComputer,
        input_data: Tensor,
        input_region_name: str,
        params: Tensor,
        params_region_name: str,
    ) -> Tensor:
        programs = [program.copy() for _ in range(input_data.shape[0])]
        inputs = [input_data[i].tolist() for i in range(input_data.shape[0])]

        if params.dim() == 1:
            params = params.reshape((1, -1))

        params_ = [params[i].tolist() for i in range(input_data.shape[0])]

        def run(
            program_instance: Program,
            single_input: List[float],
            single_parameters: List[float],
        ):
            program_instance.write_memory(
                region_name=input_region_name, value=single_input
            )
            program_instance.write_memory(
                region_name=params_region_name, value=single_parameters
            )

            qc.run(program_instance)
            bit_strings = qc.run(program_instance).readout_data.get("ro")

            return np.mean(bit_strings, 0)

        with ThreadPool(2) as pool:
            output_arrays = pool.starmap(run, zip(programs, inputs, params_))

        return torch.tensor(np.stack(output_arrays), dtype=torch.float32)


class PyQuilLayer(torch.nn.Module):
    def __init__(
        self,
        program: Program,
        qc: QuantumComputer,
        input_region_name: str,
        params_region_name: str,
        params_num: int,
        shift: float,
    ):
        super(PyQuilLayer, self).__init__()
        self.program = program
        self.qc = qc
        self.input_region_name = input_region_name
        self.params_region_name = params_region_name
        self.params_num = params_num
        self.shift = shift

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
        )


if __name__ == "__main__":
    program, params_num = QNN1.create_circuit(2)
    program.wrap_in_numshots_loop(1000)
    qc = get_qc("4q-qvm")
    executable = qc.compile(program)

    training_input = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    training_target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    model = PyQuilLayer(executable, qc, "input", "params", params_num, 0.1)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for i in range(100):
        time1 = time.time()
        pred = model(training_input)
        loss = loss_fn(pred, training_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time2 = time.time()

        print(f"loss: {loss.item():>7f} output: {pred} time: {time2 - time1}")
