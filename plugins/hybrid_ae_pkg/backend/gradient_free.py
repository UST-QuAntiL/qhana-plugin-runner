from typing import Callable

import torch
import numpy as np


def scipy_compatible_objective_function(
    parameter_values: np.ndarray,
    neural_network: torch.nn.Module,
    training_input: torch.Tensor,
    training_target: torch.Tensor,
    loss_function: Callable,
) -> float:
    if parameter_values.size != get_number_of_parameters(neural_network):
        raise ValueError(
            "Size of parameter_values does not match the number of parameters in the neural_network."
        )

    with torch.no_grad():
        set_parameters_from_1D_array(neural_network, parameter_values)
        loss = loss_function(neural_network(training_input), training_target)

    return loss


def set_parameters_from_1D_array(module: torch.nn.Module, parameter_values: np.ndarray):
    with torch.no_grad():
        offset = 0

        for param in module.parameters():
            param_size = get_tensor_size(param)
            param[:] = torch.tensor(parameter_values[offset : offset + param_size])
            offset += param_size


def get_tensor_size(tensor: torch.Tensor) -> int:
    tensor_size = 1

    for dim_size in tensor.size():
        tensor_size *= dim_size

    return tensor_size


def get_number_of_parameters(module: torch.nn.Module) -> int:
    params_num: int = 0

    for param_tensor in module.parameters():
        params_num += get_tensor_size(param_tensor)

    return params_num


if __name__ == "__main__":
    net = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1)
    )

    print(get_number_of_parameters(net))
