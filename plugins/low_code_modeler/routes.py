from http import HTTPStatus
from datetime import datetime, timezone
from uuid import uuid4
import json
from flask import render_template, send_from_directory, request
from flask_smorest import abort
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from flask.views import MethodView
from flask.helpers import url_for
from flask.wrappers import Response

from .schemas import MetadataOfModellSchema, SaveModelParamsSchema

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginMetadataSchema,
    EntryPoint,
    PluginType,
)

from .plugin import LCM_BLP, LowCodeModeler

LCM_STATE_KEY = "saved_models"


@LCM_BLP.route("/")
class PluginRootView(MethodView):
    @LCM_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @LCM_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return PluginMetadata(
            title="LCM",
            description=LowCodeModeler.instance.description,
            name=LowCodeModeler.instance.name,
            version=LowCodeModeler.instance.version,
            tags=LowCodeModeler.instance.tags,
            type=PluginType.interaction,
            entry_point=EntryPoint(
                href=url_for(f"{LCM_BLP.name}.{Process.__name__}"),
                ui_href=url_for(f"{LCM_BLP.name}.{UI.__name__}", path="index.html"),
                plugin_dependencies=[],
                data_input=[],
                data_output=[],
            ),
        )


@LCM_BLP.route("/ui/<path:path>")
class UI(MethodView):
    @LCM_BLP.response(HTTPStatus.OK)
    @LCM_BLP.require_jwt("jwt", optional=True)
    def get(self, path):
        return send_from_directory(f"{LCM_BLP.root_path}/static", path)


@LCM_BLP.route("/process/")
class Process(MethodView):
    """TODO."""

    @LCM_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return ""


@LCM_BLP.route("/models/")
class ModelListView(MethodView):

    @LCM_BLP.response(HTTPStatus.OK, MetadataOfModellSchema(many=True))
    @LCM_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Get list of all models saved"""
        plugin = LowCodeModeler.instance

        models = PluginState.get_value(plugin.name, LCM_STATE_KEY, [])
        if models is None:
            models = []
        return models

    @LCM_BLP.response(HTTPStatus.OK, MetadataOfModellSchema())
    @LCM_BLP.require_jwt("jwt", optional=True)
    def post(self):
        """Post to save a new Model"""
        plugin = LowCodeModeler.instance

        model_data = request.get_json()

        model_id = str(uuid4())
        model = {
            "id": model_data.get("id", "unnamed"),
            "name": model_data.get("name", "Unnamed Model"),
            "version": model_data.get("version", "1.0"),
            "date": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "autosave": model_data.get("autosave", False),
        }

        DataBlob.set_value(plugin.name, model_id, json.dumps(model_data).encode("utf-8"))

        models = PluginState.get_value(plugin.name, LCM_STATE_KEY, [])
        if models is None:
            models = []

        PluginState.set_value(plugin.name, LCM_STATE_KEY, [model] + models, commit=True)

        return model


@LCM_BLP.route("/models/<string:model_id>/")
class ModelView(MethodView):
    """Get a specific model file."""

    @LCM_BLP.response(HTTPStatus.OK, content_type="application/json")
    @LCM_BLP.require_jwt("jwt", optional=True)
    def get(self, model_id: str):
        """Get a specific model by its ID."""
        plugin = LowCodeModeler.instance

        model_data = DataBlob.get_value(plugin.name, model_id, default=None)
        if model_data is None:
            abort(HTTPStatus.NOT_FOUND, message=f"Model {model_id} not found")

        return Response(model_data, status=HTTPStatus.OK, content_type="application/json")

    @LCM_BLP.response(HTTPStatus.OK)
    @LCM_BLP.require_jwt("jwt", optional=True)
    def delete(self, model_id: str):
        """used to delete a model."""
        plugin = LowCodeModeler.instance

        DataBlob.delete_value(plugin.name, model_id)

        models = PluginState.get_value(plugin.name, LCM_STATE_KEY, [])
        if models is None:
            models = []

        filtered = [m for m in models if m["model_id"] != model_id]
        PluginState.set_value(plugin.name, LCM_STATE_KEY, filtered, commit=True)

        return {"success": True}
