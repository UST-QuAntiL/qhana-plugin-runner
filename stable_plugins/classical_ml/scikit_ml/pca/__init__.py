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

from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "pca"
__version__ = "v0.2.2"
_identifier = plugin_identifier(_plugin_name, __version__)


PCA_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="PCA plugin API",
    template_folder="templates",
)

sklearn_version = "1.1"


class PCA(QHAnaPluginBase):
    name = _plugin_name
    description = (
        "The PCA Plugin reduces the number of dimensions by computing the principle components.\n"
        "The new orthonormal basis consists of the k first principle components. "
        "The methods implemented here are from scikit-learn. "
        f"Currently this plugin uses scikit-learn version {sklearn_version}.\n\n"
        "The entity points should be saved in the [entity/vector](https://qhana-plugin-runner.readthedocs.io/en/latest/data-formats/examples/entities.html#entity-vector) format "
        "and they may be stored in either a csv or a json file. The ``data-creator`` plugin can generate some entity points."
    )
    version = __version__
    tags = ["preprocessing", "feature-engineering"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return PCA_BLP

    def get_requirements(self) -> str:
        return f"scikit-learn~={sklearn_version}\nplotly~=5.18.0\npandas~=1.5.0"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
