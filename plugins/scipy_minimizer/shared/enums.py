from enum import Enum


class InteractionEndpointType(Enum):
    """Type of the interaction endpoint.
    - ``minimization``: type for optimization steps
    - ``calc_loss``: type for objective function calculations
    - ``calc_grad``: type for gradient calculations
    - ``calc_loss_and_grad``: type for objective function and gradient calculations
    - ``of_pass_data``: type for passing additional data to the objective function
    """

    minimization = "minimization"
    calc_loss = "calc_loss"
    calc_grad = "calc_grad"
    calc_loss_and_grad = "calc_loss_and_grad"
    of_pass_data = "of_pass_data"
