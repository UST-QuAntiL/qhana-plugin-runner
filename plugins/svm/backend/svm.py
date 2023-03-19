from sklearn.svm import SVC
from typing import Optional, Callable, List
import numpy as np


def get_svc(
    data,
    labels,
    c=1.0,
    kernel: str | Callable[[np.array], np.array] = "rbf",
    degree=3,
    precomputed_kernel: Optional[List[List[float]]] = None,
):
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
