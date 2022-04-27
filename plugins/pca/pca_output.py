from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, SparsePCA
from sklearn.base import BaseEstimator

"""
This file prepares a dictionary with all the necessary parameters of the PCA, to do more transformations.
In other scikit-learn versions the value names could change!
"""


def PCA_output(pca: PCA) -> (dict, int):
    PCA_out = {
        "type": "normal",
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "explained_variance": pca.explained_variance_.tolist()
    }
    dim = pca.n_components_
    return PCA_out, dim


def IncrementalPCA_output(pca: IncrementalPCA) -> (dict, int):
    incrementalPCA_out = {
        "type": "incremental",
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "explained_variance": pca.explained_variance_.tolist()
    }
    dim = pca.n_components_
    return incrementalPCA_out, dim


def SparsePCA_output(pca: SparsePCA) -> (dict, int):
    SparsePCA_out = {
        "type": "sparse",
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
        "regularization_strength": pca.ridge_alpha
    }
    dim = pca.n_components_
    return SparsePCA_out, dim


def KernelPCA_output(pca: KernelPCA) -> (dict, int):
    KernelPCA_out = {
        "type": "kernel",
        "eigenvalues": pca.lambdas_.tolist(),
        "eigenvectors": pca.alphas_.tolist(),
        "kernel_centerer":
            {
                "K_fit_rows": pca._centerer.K_fit_rows_.tolist(),
                "K_fit_all": pca._centerer.K_fit_all_.tolist()
            },
        "training_data": pca.X_fit_.tolist()
    }
    # These values are for the get_kernel methode of KernelPCA.
    # This might has to be expanded/changed, if more functionality is added
    KernelPCA_out["kernel"] = pca.kernel
    KernelPCA_out["gamma"] = pca.gamma
    KernelPCA_out["degree"] = pca.degree
    KernelPCA_out["coef0"] = pca.coef0
    dim = len(pca.alphas_[0])
    return KernelPCA_out, dim


# dim = num features of output. Return dim here, since input params of the plugin can be <= 0
def pca_to_output(pca: BaseEstimator) -> (dict, int):
    if type(pca) == PCA:
        return PCA_output(pca)
    elif type(pca) == IncrementalPCA:
        return IncrementalPCA_output(pca)
    elif type(pca) == SparsePCA:
        return SparsePCA_output(pca)
    elif type(pca) == KernelPCA:
        return KernelPCA_output(pca)
    else:
        raise ValueError(f"Creating an output for object type {type(pca)} is not implemented!")
