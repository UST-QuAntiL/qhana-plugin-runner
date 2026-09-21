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
from dataclasses import asdict, dataclass
from http import HTTPStatus
from json import dumps
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
from qhana_plugin_runner.plugin_utils.attributes import (
    NUMERIC_TYPES,
    parse_attribute_metadata,
)
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
    template_folder="templates",
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
    input_range_auto: bool
    input_range_min: Optional[float]
    input_range_max: Optional[float]
    output_range_min: float
    output_range_max: float


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
        load_default=None,
        data_input_type="entity/attribute-metadata",
        data_content_types=["application/json", "application/X-lines+json", "text/csv"],
        metadata={
            "label": "Attribute Metadata URL (optional)",
            "description": "Optional metadata used to verify that selected attributes are numeric.",
            "input_type": "text",
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
    input_range_auto = ma.fields.Boolean(
        required=False,
        load_default=True,
        metadata={
            "label": "Determine input range automatically",
            "description": "Use the minimum and maximum values found in the input data (per attribute).",
            "input_type": "checkbox",
        },
    )
    input_range_min = ma.fields.Float(
        required=False,
        allow_none=True,
        load_default=None,
        metadata={
            "label": "Input range minimum",
            "description": "Manual minimum applied to every selected attribute.",
            "input_type": "number",
        },
    )
    input_range_max = ma.fields.Float(
        required=False,
        allow_none=True,
        load_default=None,
        metadata={
            "label": "Input range maximum",
            "description": "Manual maximum applied to every selected attribute.",
            "input_type": "number",
        },
    )
    output_range_min = ma.fields.Float(
        required=False,
        load_default=0.0,
        metadata={
            "label": "Output range minimum",
            "description": "Target minimum applied to every selected attribute. Common choices are [0..1], [-1..1] and [0..100].",
            "input_type": "number",
        },
    )
    output_range_max = ma.fields.Float(
        required=False,
        load_default=1.0,
        metadata={
            "label": "Output range maximum",
            "description": "Target maximum applied to every selected attribute. Common choices are [0..1], [-1..1] and [0..100].",
            "input_type": "number",
        },
    )

    @validates_schema
    def validate_parameters(self, data, **kwargs):
        errors = {}
        try:
            attributes = parse_attribute_names(data.get("attributes", ""))
        except ValueError as error:
            errors["attributes"] = [str(error)]
            attributes = []
        if not attributes and "attributes" not in errors:
            errors["attributes"] = ["At least one attribute is required."]

        if not data.get("input_range_auto", True):
            if data.get("input_range_min") is None:
                errors.setdefault("input_range_min", []).append(
                    "A minimum is required when automatic range detection is disabled."
                )
            if data.get("input_range_max") is None:
                errors.setdefault("input_range_max", []).append(
                    "A maximum is required when automatic range detection is disabled."
                )
            if (
                data.get("input_range_min") is not None
                and data.get("input_range_max") is not None
                and (
                    not math.isfinite(data["input_range_min"])
                    or not math.isfinite(data["input_range_max"])
                    or data["input_range_min"] >= data["input_range_max"]
                )
            ):
                errors["input_range_max"] = [
                    "The input maximum must be greater than the input minimum."
                ]

        output_minimum = data.get("output_range_min")
        output_maximum = data.get("output_range_max")
        if output_minimum is None:
            errors.setdefault("output_range_min", []).append(
                "An output minimum is required."
            )
        if output_maximum is None:
            errors.setdefault("output_range_max", []).append(
                "An output maximum is required."
            )
        if (
            output_minimum is not None
            and output_maximum is not None
            and (
                not math.isfinite(output_minimum)
                or not math.isfinite(output_maximum)
                or output_minimum >= output_maximum
            )
        ):
            errors["output_range_max"] = [
                "The output maximum must be greater than the output minimum."
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
        # Do not show validation errors when the form is opened without input.
        initial_errors = {} if not request.args else errors
        return self.render(request.args, initial_errors, False)

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
        return Response(
            render_template(
                "normalization.html",
                name=Normalization.instance.name,
                version=Normalization.instance.version,
                schema=schema,
                valid=valid,
                values=data,
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
            task_name=calculation_task.name, parameters=dumps(arguments)
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


def parse_attribute_names(value: str) -> List[str]:
    """Parse a newline-separated list of unique entity attribute names."""
    names = [line.strip() for line in value.splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError("The attribute list must not contain duplicates.")
    return names


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
    attribute_metadata: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Normalize selected scalar numeric attributes while preserving entities."""
    if not entities:
        raise ValueError("The input entity file must not be empty.")
    if not params.attributes:
        raise ValueError("At least one attribute must be selected.")
    if not params.input_range_auto and (
        params.input_range_min is None
        or params.input_range_max is None
        or not math.isfinite(params.input_range_min)
        or not math.isfinite(params.input_range_max)
        or params.input_range_min >= params.input_range_max
    ):
        raise ValueError(
            "The manual input range must have a minimum smaller than its maximum."
        )
    if (
        not math.isfinite(params.output_range_min)
        or not math.isfinite(params.output_range_max)
        or params.output_range_min >= params.output_range_max
    ):
        raise ValueError("The output range must have a minimum smaller than its maximum.")

    if attribute_metadata is not None:
        for attribute in params.attributes:
            metadata = attribute_metadata.get(attribute)
            if metadata is None or metadata.attribute_type.lower() not in NUMERIC_TYPES:
                raise ValueError(
                    f"Attribute '{attribute}' is not declared as numeric metadata."
                )
            if metadata.multiple:
                raise ValueError(f"Attribute '{attribute}' must contain scalar values.")

    columns: Dict[str, List[float]] = {attribute: [] for attribute in params.attributes}
    for entity in entities:
        if "ID" not in entity:
            raise ValueError("Every entity must contain an ID attribute.")
        for attribute in params.attributes:
            if attribute not in entity:
                raise ValueError(
                    f"Entity '{entity['ID']}' has no value for attribute '{attribute}'."
                )
            columns[attribute].append(
                _parse_numeric_value(entity["ID"], attribute, entity[attribute])
            )

    ranges = {}
    for attribute, values in columns.items():
        if params.input_range_auto:
            minimum, maximum = min(values), max(values)
        else:
            minimum, maximum = params.input_range_min, params.input_range_max
        if minimum == maximum:
            raise ValueError(f"Attribute '{attribute}' has a constant input range.")
        ranges[attribute] = (minimum, maximum)

    normalized = []
    for index, entity in enumerate(entities):
        result = dict(entity)
        for attribute in params.attributes:
            input_minimum, input_maximum = ranges[attribute]
            value = columns[attribute][index]
            result[attribute] = params.output_range_min + (
                (value - input_minimum)
                * (params.output_range_max - params.output_range_min)
                / (input_maximum - input_minimum)
            )
        normalized.append(result)
    return normalized


@CELERY.task(name=f"{Normalization.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    task_data = ProcessingTask.get_by_id(db_id)
    params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    with open_url(params.entities_url) as response:
        input_mimetype = get_mimetype(response)
        entities = list(ensure_dict(load_entities(response, input_mimetype)))

    metadata = None
    metadata_url = params.attribute_metadata_url
    if metadata_url:
        with open_url(metadata_url) as response:
            metadata_entities = ensure_dict(
                load_entities(response, get_mimetype(response))
            )
            metadata = parse_attribute_metadata(metadata_entities)

    normalized = normalize_entities(
        entities=entities,
        params=params,
        attribute_metadata=metadata,
    )

    output_attributes = list(entities[0].keys())
    with SpooledTemporaryFile(mode="w+") as output:
        save_entities(
            entities=normalized,
            file_=output,
            mimetype=input_mimetype,
            attributes=output_attributes if input_mimetype == "text/csv" else None,
        )
        output.seek(0)
        extension = {
            "application/json": ".json",
            "application/X-lines+json": ".jsonl",
            "text/csv": ".csv",
        }.get(input_mimetype, ".data")
        STORE.persist_task_result(
            db_id,
            output,
            f"normalized_entities{extension}",
            "entity/list",
            input_mimetype,
        )

    return "Numeric values normalized successfully."
