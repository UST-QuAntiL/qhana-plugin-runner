from typing import Optional

import numpy as np
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from . import RidgeLoss

TASK_LOGGER = get_task_logger(__name__)


def ridge_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """
    Calculate the ridge loss given weights, features, target, and alpha.

    Args:
        w: Weights
        X: Features
        y: Target
        alpha: Ridge regularization parameter

    Returns:
        The calculated ridge loss.
    """
    y_pred = np.dot(X, w)
    mse = np.mean((y - y_pred) ** 2)
    ridge_penalty = alpha * np.sum(w**2)
    return mse + ridge_penalty
