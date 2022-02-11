from typing import Tuple
import math

import torch

from plugins.hybrid_ae_pkg.backend.quantum.qiskit.autoencoder import QuantumAutoencoder


# TODO: measure performance
class QAEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qae: QuantumAutoencoder,
        input_values: torch.Tensor,
        encoder_params: torch.Tensor,
        shift: float,
    ) -> torch.Tensor:
        """
        Forward pass of the Quantum Autoencoder.
        :param qae: QuantumAutoencoder instance that should be run
        :param ctx: context
        :param input_values: size: [instance_cnt, qubit_cnt]
        :param encoder_params: size: [encoder_params_cnt]
        :param decoder_params: size: [decoder_params_cnt]
        :param shift: shift for the finite differences calculation
        :return: Output from the QuantumAutoencoder. size: [instance_cnt, qubit_cnt]
        """
        ctx.qae = qae
        ctx.shift = shift
        ctx.save_for_backward(input_values, encoder_params)

        return qae.run(
            input_values,
            encoder_params.reshape(1, -1).repeat(input_values.shape[0], 1),
            1024,
        )

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, torch.Tensor, None]:
        """
        Backward pass of the Quantum Autoencoder. Approximates the gradients.
        :param ctx: context
        :param grad_output: Gradient with respect to the output of the Quantum Autoencoder. size: [
        :return:
        """
        qae: QuantumAutoencoder = ctx.qae
        input_values: torch.Tensor
        encoder_params: torch.Tensor
        shift: float = ctx.shift

        input_values, encoder_params = ctx.saved_tensors

        # gradient approximation with central differences
        encoder_params_extended = encoder_params.reshape(1, -1).repeat(2, 1)

        input_grad = torch.zeros(input_values.shape)

        for i in range(input_values.shape[0]):
            for j in range(input_values.shape[1]):
                right_shift = input_values[i].clone().detach()
                right_shift[j] += shift
                left_shift = input_values[i].clone().detach()
                left_shift[j] -= shift

                shifted_input = torch.stack([right_shift, left_shift], dim=0)
                output = qae.run(shifted_input, encoder_params_extended, 1000)

                derivatives = (output[0] - output[1]) / (2.0 * shift)
                combined_derivative = (derivatives * grad_output).sum()

                input_grad[i, j] = combined_derivative

        encoder_grad = torch.zeros(encoder_params.shape)

        shifted_encoder_params = []

        for i in range(encoder_params.shape[0]):
            for j in range(input_values.shape[0]):
                right_shift = encoder_params.clone().detach()
                right_shift[i] += shift
                left_shift = encoder_params.clone().detach()
                left_shift[i] -= shift

                shifted_encoder_params.append(right_shift)
                shifted_encoder_params.append(left_shift)

        extended_input_values = input_values.repeat_interleave(2, 0).repeat(
            (encoder_params.shape[0], 1)
        )
        stacked_params = torch.stack(shifted_encoder_params, dim=0)
        output = qae.run(extended_input_values, stacked_params, 1000)

        for i in range(encoder_params.shape[0]):
            for j in range(input_values.shape[0]):
                derivatives = (
                    output[2 * (i * input_values.shape[0] + j)]
                    - output[2 * (i * input_values.shape[0] + j) + 1]
                ) / (2.0 * shift)
                combined_derivative = (derivatives * grad_output).sum()

                encoder_grad[i] += combined_derivative

        return None, input_grad, encoder_grad, None


class QAEModule(torch.nn.Module):
    def __init__(self, qae: QuantumAutoencoder):
        super(QAEModule, self).__init__()
        self.qae = qae
        self.encoder_params = torch.nn.Parameter(
            torch.rand(qae.get_encoder_param_cnt()) * 2.0 * math.pi
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QAEFunction.apply(self.qae, x, self.encoder_params, math.pi / 4.0)
