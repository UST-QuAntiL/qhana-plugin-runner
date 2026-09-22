# Copyright 2026 QHAna plugin runner contributors.
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
import math
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

from celery.utils.log import get_task_logger

import marshmallow as ma
from marshmallow import validates_schema
from celery.canvas import chain
from flask import Response, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import NUMERIC_TYPES, AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.requests import get_mimetype, open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "numeric-value-normalization"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)
_description = r"""
Normalizes selected numeric entity attributes from an input range to an output range.

For each selected value $x$, the normalized value $y$ is calculated as:

$$
y = y_{min} + \frac{x - x_{min}}{x_{max} - x_{min}} \cdot (y_{max} - y_{min})
$$

The input range can be determined automatically for each attribute or supplied manually.
""".strip()


NORMALIZATION_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description=_description,
)


class Normalization(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = _description
    tags = ["preprocessing"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return NORMALIZATION_BLP


@dataclass
class InputParameters:
    entities_url: str
    attribute_metadata_url: Optional[str]
    attributes: List[str]
    input_range_min: Optional[float]
    input_range_max: Optional[float]
    output_range_min: float
    output_range_max: float


class NullableFloat(ma.fields.Float):
    """Float field that treats an empty string form value as None."""

    def deserialize(self, value, attr=None, data=None, **kwargs):
        if value == "":
            value = None
        return super().deserialize(value, attr, data, **kwargs)


class InputParametersSchema(FrontendFormBaseSchema):
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/list",
        data_content_types=["application/json", "application/X-lines+json", "text/csv"],
        metadata={
            "label": "Entities URL",
            "description": "URL to a file with entities.",
            "input_type": "text",
        },
    )
    attribute_metadata_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/attribute-metadata",
        data_content_types=["application/json", "application/X-lines+json", "text/csv"],
        metadata={
            "label": "Attribute Metadata URL (optional)",
            "description": "Optional metadata used to verify that selected attributes are numeric. If not provided, the metadata will be fetched from the X-Attribute-Metadata header of the entities file, if present.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "post",
        },
    )
    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Attributes",
            "description": "Numeric attributes to normalize, one attribute per line.",
            "input_type": "textarea",
        },
    )
    input_range_min = NullableFloat(
        required=False,
        allow_none=True,
        metadata={
            "label": "Input range minimum",
            "description": "Manual minimum applied to every attribute. If empty, the minimum of each attribute is used.",
            "input_type": "number",
        },
    )
    input_range_max = NullableFloat(
        required=False,
        allow_none=True,
        metadata={
            "label": "Input range maximum",
            "description": "Manual maximum applied to every attribute. If empty, the maximum of each attribute is used.",
            "input_type": "number",
        },
    )
    output_range_min = ma.fields.Float(
        required=True,
        metadata={
            "label": "Output range minimum",
            "description": "Target minimum applied to every selected attribute. Common choices are [0..1], [-1..1] and [0..100].",
            "input_type": "number",
        },
    )
    output_range_max = ma.fields.Float(
        required=True,
        metadata={
            "label": "Output range maximum",
            "description": "Target maximum applied to every selected attribute. Common choices are [0..1], [-1..1] and [0..100].",
            "input_type": "number",
        },
    )

    @validates_schema
    def validate_parameters(self, data, **kwargs):
        errors = {}

        input_minimum = data.get("input_range_min")
        input_maximum = data.get("input_range_max")

        if input_minimum is not None and input_maximum is not None:
            if not math.isfinite(input_minimum) or not math.isfinite(input_maximum):
                errors["input_range_max"] = [
                    "The input minimum and maximum must be finite numbers."
                ]
            elif input_minimum >= input_maximum:
                errors["input_range_max"] = [
                    "The input maximum must be greater than the input minimum."
                ]

        output_minimum = data.get("output_range_min")
        output_maximum = data.get("output_range_max")
        if output_minimum is None or output_maximum is None:
            errors["output_range_max"] = ["An output minimum and maximum is required."]
        elif not math.isfinite(output_minimum) or not math.isfinite(output_maximum):
            errors["output_range_max"] = [
                "The output minimum and maximum must be finite numbers."
            ]
        elif output_minimum >= output_maximum:
            errors["output_range_max"] = [
                "The output maximum must be greater than the output minimum and."
            ]

        if errors:
            raise ma.ValidationError(errors)

    @ma.post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@NORMALIZATION_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @NORMALIZATION_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @NORMALIZATION_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return PluginMetadata(
            title="Numeric Value Normalization",
            description=Normalization.instance.description,
            name=Normalization.instance.name,
            version=Normalization.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{NORMALIZATION_BLP.name}.ProcessView"),
                ui_href=url_for(f"{NORMALIZATION_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/list",
                        content_type=[
                            "application/json",
                            "application/X-lines+json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="entitiesUrl",
                    ),
                    InputDataMetadata(
                        data_type="entity/attribute-metadata",
                        content_type=[
                            "application/json",
                            "application/X-lines+json",
                            "text/csv",
                        ],
                        required=False,
                        parameter="attributeMetadataUrl",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/list",
                        content_type=[
                            "application/json",
                            "application/X-lines+json",
                            "text/csv",
                        ],
                        required=True,
                    )
                ],
            ),
            tags=Normalization.instance.tags,
        )


@NORMALIZATION_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Normalization plugin."""

    @NORMALIZATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Normalization plugin."
    )
    @NORMALIZATION_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @NORMALIZATION_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @NORMALIZATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the Normalization plugin."
    )
    @NORMALIZATION_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @NORMALIZATION_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        schema = InputParametersSchema()
        fields = schema.fields
        default_values = {
            fields["output_range_min"].data_key: 0.0,
            fields["output_range_max"].data_key: 1.0,
        }
        default_values.update(data)
        return Response(
            render_template(
                "simple_template.html",
                name=Normalization.instance.name,
                version=Normalization.instance.version,
                schema=schema,
                valid=valid,
                values=default_values,
                errors=errors,
                process=url_for(f"{NORMALIZATION_BLP.name}.ProcessView"),
            )
        )


@NORMALIZATION_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @NORMALIZATION_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @NORMALIZATION_BLP.response(HTTPStatus.SEE_OTHER)
    @NORMALIZATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=InputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = calculation_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


TASK_LOGGER = get_task_logger(__name__)


def _load_entities(
    entities_url: str, attribute_metadata_url: Optional[str] = None
) -> Tuple[List[dict], Dict[str, Any], str]:
    """Load entities and attribute metadata from the given URLs."""
    with open_url(entities_url) as entities_data:
        mimetype = get_mimetype(entities_data)
        attribute_metadata: dict[str, Any] = {}

        if attribute_metadata_url is None:
            attribute_metadata_url = entities_data.headers.get("X-Attribute-Metadata")

        if attribute_metadata_url is not None:
            with open_url(attribute_metadata_url) as attribute_metadata_file:
                attribute_metadata = {
                    attr_meta["ID"]: AttributeMetadata.from_dict(attr_meta)
                    for attr_meta in ensure_dict(
                        load_entities(
                            attribute_metadata_file,
                            get_mimetype(attribute_metadata_file),
                        )
                    )
                }

        entities = list(
            ensure_dict(load_entities(entities_data, mimetype), attribute_metadata)
        )
        if not entities:
            raise ValueError("The input entity file must not be empty.")

    return entities, attribute_metadata, mimetype


def _parse_numeric_value(entity_id: Any, attribute: str, value: Any) -> float:
    if isinstance(value, bool) or isinstance(value, (list, tuple, set, dict)):
        raise ValueError(
            f"Entity '{entity_id}' has a non-scalar value for attribute '{attribute}'."
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Entity '{entity_id}' has a non-numeric value for attribute '{attribute}'."
        ) from None
    if not math.isfinite(number):
        raise ValueError(
            f"Entity '{entity_id}' has a non-finite value for attribute '{attribute}'."
        )
    return number


def normalize_entities(
    entities: Sequence[Mapping[str, Any]],
    params: InputParameters,
    attribute_metadata: Optional[Mapping[str, AttributeMetadata]] = None,
) -> List[Dict[str, Any]]:
    """Normalize selected scalar numeric attributes while preserving entities."""
    selected_attributes = params.attributes.splitlines()
    print(selected_attributes)
    print(params.attributes)
    output_min = params.output_range_min
    output_max = params.output_range_max
    output_range = output_max - output_min
    if attribute_metadata is not None:
        for attribute in selected_attributes:
            metadata: AttributeMetadata | None = attribute_metadata.get(attribute)
            print(metadata)
            if metadata is None or metadata.description not in NUMERIC_TYPES:
                raise ValueError(
                    f"Attribute '{attribute}' is not declared as numeric metadata."
                )
            if metadata.multiple:
                raise ValueError(f"Attribute '{attribute}' must contain scalar values.")

    columns: Dict[str, List[float]] = {attribute: [] for attribute in selected_attributes}
    for entity in entities:
        if "ID" not in entity:
            raise ValueError("Every entity must contain an ID attribute.")
        for attribute in selected_attributes:
            if attribute not in entity:
                raise ValueError(
                    f"Entity '{entity['ID']}' has no value for attribute '{attribute}'."
                )
            columns[attribute].append(
                _parse_numeric_value(entity["ID"], attribute, entity[attribute])
            )

    input_ranges = {}
    for attribute, values in columns.items():
        input_min = params.input_range_min
        if input_min is None:
            input_min = min(values)
        input_max = params.input_range_max
        if input_max is None:
            input_max = max(values)
        input_ranges[attribute] = (input_min, input_max)

    normalized = []
    for index, entity in enumerate(entities):
        result = dict(entity)
        for attribute in selected_attributes:
            input_min, input_max = input_ranges[attribute]
            value = columns[attribute][index]
            scaled = output_min + (
                (value - input_min) * output_range / (input_max - input_min)
            )
            # Values outside a manual input range must not leave the output range.
            result[attribute] = min(
                max(scaled, params.output_range_min), params.output_range_max
            )
        normalized.append(result)
    return normalized


@CELERY.task(name=f"{Normalization.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    TASK_LOGGER.info("Starting numeric value normalization task with db_id=%s", db_id)
    task_data = ProcessingTask.get_by_id(db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: InputParameters = InputParametersSchema().loads(task_data.parameters)
    entities, attribute_metadata, mimetype = _load_entities(
        params.entities_url, params.attribute_metadata_url
    )

    normalized = normalize_entities(
        entities=entities,
        params=params,
        attribute_metadata=attribute_metadata,
    )

    output_attributes = list(entities[0].keys())
    with SpooledTemporaryFile(mode="w+") as output:
        save_entities(
            entities=normalized,
            file_=output,
            mimetype=mimetype,
            attributes=output_attributes if mimetype == "text/csv" else None,
        )
        output.seek(0)
        extension = Path(urlparse(params.entities_url).path).suffix
        STORE.persist_task_result(
            db_id,
            output,
            f"normalized_entities{extension}",
            "entity/list",
            mimetype,
        )

    return "Numeric values normalized successfully."
