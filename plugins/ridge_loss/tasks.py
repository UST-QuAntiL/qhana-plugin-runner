# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from celery.utils.log import get_task_logger

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