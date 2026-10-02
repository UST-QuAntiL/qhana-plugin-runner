from typing import ClassVar, Optional

from flask import Blueprint, Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_name = "sql-editor"
_version = "v0.1.0"


SQL_BLP = SecurityBlueprint(
    plugin_identifier(_name, _version),
    __name__,
    description="SQL Plugin using DuckDB.",
    template_folder="templates",
)


class SQLEditor(QHAnaPluginBase):
    name = _name
    version = _version
    description = "A plugin to use SQL for data loading or data cleaning."
    tags = ["sql", "duckdb", "preprocessing", "data-cleaning"]

    instance: ClassVar["SQLEditor"]

    _blueprint: Optional[Blueprint] = None

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def init_app(self, app: Flask):
        super().init_app(app)

    def get_requirements(self) -> str:
        return "duckdb~=1.4.2"

    def get_api_blueprint(self):
        if not self._blueprint:
            self._blueprint = SQL_BLP
            return SQL_BLP
        return self._blueprint
