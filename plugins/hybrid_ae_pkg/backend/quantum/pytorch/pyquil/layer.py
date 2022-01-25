from typing import Any, Tuple

from pyquil import Program
from pyquil.api import QuantumComputer
import torch
from torch import Tensor
from torch.autograd import Function
import numpy as np


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

        return PyquilLayer._calc_output(
            program, qc, input_data, input_region_name, params, params_region_name
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
                output = PyquilLayer._calc_output(
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
        output = PyquilLayer._calc_output(
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
        output_arrays = []

        # TODO: parallel execution of circuits
        for i in range(input_data.shape[0]):
            program.write_memory(region_name=input_region_name, value=input_data[i])
            program.write_memory(region_name=params_region_name, value=params[i])

            qc.run(program)
            bit_strings = qc.run(program).readout_data.get("ro")
            output_arrays.append(np.mean(bit_strings, 0))

        return torch.tensor(np.stack(output_arrays), dtype=torch.float32)
