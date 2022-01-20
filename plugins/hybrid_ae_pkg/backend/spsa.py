# TODO: implement SPSA as PyTorch optimizer
# TODO: make it compatible with e.g. the Adam optimizer
# https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer
# https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.step
import math
from typing import Callable

import numpy as np
from numpy.random import default_rng


class SPSA:
    def __init__(
        self,
        obj_func: Callable[[np.ndarray], float],
        k: int = 0,
        theta: np.ndarray = None,
        a: float = 0,
        c: float = 1,
        capital_a: float = 0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        max_iter: int = 1000,
        learning_rate: float = 0.1,
    ):
        self.obj_func = obj_func  # function for which the parameters will be optimized
        self.k = k  # counter
        self.theta = theta  # parameters that will be optimized
        self.a = a  # hyperparameter
        self.c = c  # hyperparameter
        self.capital_a = capital_a  # hyperparameter
        self.alpha = alpha  # hyperparameter
        self.gamma = gamma  # hyperparameter
        self.max_iter = max_iter  # hyperparameter
        self.learning_rate = learning_rate  # hyperparameter
        self.rng = default_rng()  # pseudo random number generator

    def generate_bernoulli_like(self, a: np.ndarray) -> np.ndarray:
        return self.rng.choice([-1, 1], a.shape)

    def gain_sequence_a(self):
        return self.a / math.pow(self.capital_a + self.k + 1, self.alpha)

    def gain_sequence_c(self):
        return self.c / math.pow(self.k + 1, self.gamma)

    def approximate_gradient(self) -> np.ndarray:
        perturbation = self.generate_bernoulli_like(self.theta)
        c_k = self.gain_sequence_c()

        approx_grad = (
            self.obj_func(self.theta + c_k * perturbation)
            - self.obj_func(self.theta - c_k * perturbation)
        ) / (2 * c_k)
        approx_grad = (1 / perturbation) * approx_grad

        return approx_grad

    def update_theta(self):
        self.theta = self.theta - self.gain_sequence_a() * self.approximate_gradient()
        self.k += 1

    def approximate_good_hyperparameters(self, obj_iter: int, grad_iter: int):
        """
        The hyperparameters alpha, gamma, max_iter and learning rate, need to be set already. Approximates good values
        for the hyperparameters c, a and A using the guidelines described in J. C. Spall,
        “Implementation of the simultaneous perturbation algorithm for stochastic optimization,” IEEE Transactions on
        Aerospace and Electronic Systems, vol. 34, no. 3, pp. 817–823, Jul. 1998, doi: 10.1109/7.705889.

        @param obj_iter: number of function evaluations that will be executed
        @param grad_iter: number of gradient approximations that will be executed
        @return: nothing
        """
        # evaluate the objective function a few times at the same point
        obj_values = []

        for i in range(obj_iter):
            obj_values.append(self.obj_func(self.theta))

        std = float(np.std(obj_values))
        self.c = std  # set hyperparameter c equal to the standard deviation of the objective values
        self.c = max(
            self.c, 0.00001
        )  # makes sure that c is not too small to avoid numerical issues

        self.capital_a = (
            0.1 * self.max_iter
        )  # set A to be 10% of the maximum number of iterations

        # approximate magnitude of the approximated gradients
        avg_mag = 0.0

        for i in range(grad_iter):
            avg_mag += self.approximate_gradient().mean()

        avg_mag /= grad_iter

        self.a = (
            self.learning_rate * math.pow(self.capital_a + 1, self.alpha)
        ) / avg_mag  # set hyperparameter a


if __name__ == "__main__":
    rng = default_rng()

    def func(x: np.ndarray) -> float:
        return x[0] * 3.9 + x[1] * 2.7 + 42 + 50 * rng.random()

    spsa = SPSA(func, theta=np.ones((3,)), max_iter=1000, learning_rate=0.1)
    spsa.approximate_good_hyperparameters(50, 50)
    # print(spsa.generate_bernoulli_like(np.zeros((1,))))
    # print(spsa.approximate_gradient())

    for i in range(1000):
        spsa.update_theta()
        print(func(spsa.theta))
