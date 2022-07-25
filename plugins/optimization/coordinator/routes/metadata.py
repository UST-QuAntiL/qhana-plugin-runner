from http import HTTPStatus

from flask.helpers import url_for
from flask.views import MethodView

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    DataMetadata,
)
from .optimizer import OptimSetupProcess, OptimSelectionUI
from .. import OPTI_COORD_BLP, OptimizationCoordinator


@OPTI_COORD_BLP.route("/")
class MetadataView(MethodView):
    """Plugin metadata resource."""

    @OPTI_COORD_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=OptimizationCoordinator.instance.name,
            description=OPTI_COORD_BLP.description,
            name=OptimizationCoordinator.instance.identifier,
            version=OptimizationCoordinator.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(
                    f"{OPTI_COORD_BLP.name}.{OptimSetupProcess.__name__}"
                ),  # URL for the first process endpoint
                ui_href=url_for(
                    f"{OPTI_COORD_BLP.name}.{OptimSelectionUI.__name__}"
                ),  # URL for the first micro frontend endpoint
                interaction_endpoints=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    ),
                ],
            ),
            tags=[],
        )
