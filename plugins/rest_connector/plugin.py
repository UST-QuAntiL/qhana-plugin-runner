from typing import ClassVar, Optional

from flask import Blueprint, Flask

from qhana_plugin_runner.util.plugins import QHAnaPluginBase


class RESTConnector(QHAnaPluginBase):
    name = "rest-connector"
    version = "v0.1.0"
    description = "A plugin to integrate REST APIs as QHAna plugins."
    tags = ["rest"]

    instance: ClassVar["RESTConnector"]

    _blueprint: Optional[Blueprint] = None

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        if not self._blueprint:
            from .api.blueprint import REST_CONN_BLP

            self._blueprint = REST_CONN_BLP
            return REST_CONN_BLP
        return self._blueprint

    def get_requirements(self) -> str:
        return "prance~=23.6\nopenapi-spec-validator\nlangchain==0.0.336\nopenai==1.3.0"
