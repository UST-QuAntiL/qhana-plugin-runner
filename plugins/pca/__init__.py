from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "pca"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


PCA_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="PCA plugin API",
)

sklearn_version = "0.24.2"


class PCA(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "This plugin groups the data into different clusters, with the help of quantum algorithms.\n" \
                  "Currently there are four implemented algorithms. Destructive interference and negative rotation are from [0], " \
                  "positive correlation is from [1] and state preparation is from a previous colleague.\n\n" \
                  "Source:\n" \
                  "[0] S. Khan and A. Awan and G. Vall-Llosera. K-Means Clustering on Noisy Intermediate Scale Quantum Computers.arXiv. <a href=\"https://doi.org/10.48550/ARXIV.1909.12183\">https://doi.org/10.48550/ARXIV.1909.12183</a>\n" \
                  "[1] https://towardsdatascience.com/quantum-machine-learning-distance-estimation-for-k-means-clustering-26bccfbfcc76"
    # description = "Reduces number of dimensions. (New ONB are the d first principle components)"
    tags = ["dimension-reduction"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return PCA_BLP

    def get_requirements(self) -> str:
        return f"scikit-learn~={sklearn_version}\n" \
               "plotly~=5.3.1\n" \
               "pandas~=1.4.1"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
