# Copyright 2022 QHAna plugin runner contributors.
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

from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, SparsePCA
from sklearn.base import BaseEstimator

from celery.utils.log import get_task_logger


TASK_LOGGER = get_task_logger(__name__)

"""
This file prepares a dictionary with all the necessary parameters of the PCA, to do more transformations.
In other scikit-learn versions the value names could change!
"""


def pca_output(pca: PCA) -> (dict, int):
    pca_out = {
        "type": "normal",
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
    }
    dim = pca.n_components_
    return pca_out, dim


def incremental_pca_output(pca: IncrementalPCA) -> (dict, int):
    incremental_pca_out = {
        "type": "incremental",
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
    }
    dim = pca.n_components_
    return incremental_pca_out, dim


def sparse_pca_output(pca: SparsePCA) -> (dict, int):
    sparse_pca_out = {
        "type": "sparse",
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "regularization_strength": pca.ridge_alpha,
    }
    dim = pca.n_components_
    return sparse_pca_out, dim


def kernel_pca_output(pca: KernelPCA) -> (dict, int):
    kernel_pca_out = {
        "type": "kernel",
        "eigenvalues": pca.eigenvalues_.tolist(),
        "eigenvectors": pca.eigenvectors_.tolist(),
        "kernel_centerer": {
            "K_fit_rows": pca._centerer.K_fit_rows_.tolist(),
            "K_fit_all": pca._centerer.K_fit_all_.tolist(),
        },
        "training_data": pca.X_fit_.tolist(),
    }
    # These values are for the get_kernel methode of KernelPCA.
    # This might has to be expanded/changed, if more functionality is added
    kernel_pca_out["kernel"] = pca.kernel
    kernel_pca_out["gamma"] = pca.gamma
    kernel_pca_out["degree"] = pca.degree
    kernel_pca_out["coef0"] = pca.coef0
    dim = len(pca.eigenvectors_[0])
    return kernel_pca_out, dim


# dim = num features of output. Return dim here, since input params of the plugin can be <= 0
def pca_to_output(pca: BaseEstimator) -> (dict, int):
    """
    This method takes a fitted pca by sklearn as an input and converts its state into a dictionary inorder to save it.
    Additionally, it returns the dimensionality of a transformed data point.
    :param pca: one of sklearns pca types, e.g. PCA or KernelPCA
    :return: (dict, int) a dictionary to save the current state of the PCA and an integer equal to the output's number of dimensions
    """
    if type(pca) == PCA:
        return pca_output(pca)
    elif type(pca) == IncrementalPCA:
        return incremental_pca_output(pca)
    elif type(pca) == SparsePCA:
        return sparse_pca_output(pca)
    elif type(pca) == KernelPCA:
        return kernel_pca_output(pca)
    else:
        raise ValueError(
            f"Creating an output for object type {type(pca)} is not implemented!"
        )