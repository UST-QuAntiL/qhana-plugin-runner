from enum import Enum


class InteractionEndpointType(Enum):
    """Type of the interaction endpoint.
    - ``optimization-step``: type for optimization steps
    - ``objective_function_calc``: type for objective function calculations
    - ``of-pass-data``: type for passing additional data to the objective function
    """

    minimization_step = "minimization-step"
    objective_function_calc = "objective-function_calc"
    objective_function_gradient = "objective-function_gradient"
    of_loss_and_grad = "of-loss-and-grad"
    of_pass_data = "of-pass-data"
