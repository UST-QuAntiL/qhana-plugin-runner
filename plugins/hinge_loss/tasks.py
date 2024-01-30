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


def hinge_loss(w, X, y, C=1.0):
    """
    Hinge loss function for binary classification.

    Parameters
    ----------
    w : 1-D array
        Weight vector.

    X : 2-D array
        Input data, shape (n_samples, n_features).

    y : 1-D array
        Output data, shape (n_samples, ).

    C : float
        Regularization parameter.

    Returns
    -------
    float
        The hinge loss.
    """
    n_samples, _ = X.shape
    loss = 0.0
    for i in range(n_samples):
        score = np.dot(w, X[i])
        loss += max(0, 1 - y[i] * score)
    loss = C * loss / n_samples  # regularization term
    loss += 0.5 * np.dot(w, w)  # l2 regularization
    return loss
