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

from sklearn.svm import SVC
from typing import Optional, Callable, List
import numpy as np


def get_svc(
    data: np.array,
    labels: List,
    c: float = 1.0,
    kernel: str | Callable[[np.array], np.array] = "rbf",
    degree: int = 3,
    precomputed_kernel: Optional[List[List[float]]] = None,
) -> SVC:
    """
    train classical support vector classifier with the given parameters and data

    data: training data
    labels: training labels
    c: regularization parameter. The lower C the stronger the regularization (float)
    kernel: kernel type used for the support vector machine (string)
    degree: defree of the polynomial kernel function ('poly'). Ignored by all oterh kernels (int)
    """

    csvc = SVC(
        C=c,
        kernel=kernel,
        degree=degree,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        max_iter=-1,
    )
    if kernel == "precomputed":
        csvc.fit(precomputed_kernel, labels)
    else:
        csvc.fit(data, labels)

    return csvc
