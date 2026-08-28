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


from importlib import import_module

import pytest
from marshmallow import EXCLUDE, ValidationError

from router.routes import INPUT_FIELD_GROUPS
from router.schemas import (
    MAPPING_PLUGIN,
    NONE_PLUGIN,
    ONE_HOT_PLUGIN,
    PIPELINE_FIELD_PREFIX,
    PIPELINE_OPTIONS,
    PIPELINE_PLUGINS,
    WU_PALMER_PLUGIN,
    DistanceMetricEnum,
    InputParameters,
    InputParametersSchema,
    MetricEnum,
    MissingDataHandling,
    PCATypeEnum,
    RoutingStepParametersSchema,
    SolverEnum,
    TransformersEnum,
)
from router.tests.data import ENTITIES_URL, TAXONOMIES_URL, router_payload


def _frontend_schema(schema_class):
    """Schema instance as it is used by the micro frontend endpoints."""
    return schema_class(partial=True, unknown=EXCLUDE, validate_errors_as_result=True)


# --- INPUT PARAMETERS: HAPPY PATH ---


def test_valid_payload_loads_successfully():
    result = InputParametersSchema().load(router_payload())
    assert isinstance(result, InputParameters)
    assert result.entities_url == ENTITIES_URL
    assert result.taxonomies_zip_url == TAXONOMIES_URL
    assert result.mds_dimensions == 2
    assert result.distance_metric is DistanceMetricEnum.euclidean
    assert result.transformer is TransformersEnum.linear_inverse
    assert result.metric is MetricEnum.metric_mds
    assert result.missing_data_handling is MissingDataHandling.mean
    assert result.pca_type is PCATypeEnum.normal
    assert result.solver is SolverEnum.auto


def test_optional_fields_have_defaults():
    """Every checkbox defaults to off, so an unchecked form is still valid."""
    payload = router_payload()
    for key in (
        "includeIntermediateResultsInOutput",
        "rootIsPartOfHierarchy",
        "concatOutput",
        "reduceDimensions",
        "outputFormat",
    ):
        del payload[key]

    result = InputParametersSchema().load(payload)

    assert result.include_intermediate_results_in_output is False
    assert result.root_is_part_of_hierarchy is False
    assert result.concat_output is False
    assert result.reduce_dimensions is False
    assert result.output_format == "csv"


def test_dump_load_roundtrip_matches_process_view():
    """``ProcessView`` dumps the parameters and the task loads them again."""
    schema = InputParametersSchema()
    params = schema.load(router_payload(concatOutput=True, reduceDimensions=True))

    reloaded = schema.loads(schema.dumps(params))

    assert reloaded.entities_url == params.entities_url
    assert reloaded.distance_metric is params.distance_metric
    assert reloaded.transformer is params.transformer
    assert reloaded.metric is params.metric
    assert reloaded.missing_data_handling is params.missing_data_handling
    assert reloaded.pca_type is params.pca_type
    assert reloaded.solver is params.solver
    assert reloaded.concat_output is True
    assert reloaded.reduce_dimensions is True
    assert reloaded.output_format == params.output_format


# --- INPUT PARAMETERS: VALIDATION ---


def test_missing_required_fields_rejected():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load({})
    assert "entitiesUrl" in exc.value.messages
    assert "entitiesMetadataUrl" in exc.value.messages
    assert "taxonomiesZipUrl" in exc.value.messages
    assert "transformer" in exc.value.messages
    assert "distanceMetric" in exc.value.messages
    assert "mdsDimensions" in exc.value.messages
    assert "pcaDimensions" in exc.value.messages


@pytest.mark.parametrize(
    "field", ["entitiesUrl", "entitiesMetadataUrl", "taxonomiesZipUrl"]
)
def test_invalid_urls_rejected(field):
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(router_payload(**{field: "not-a-url"}))
    assert field in exc.value.messages


@pytest.mark.parametrize(
    "field,value",
    [
        ("distanceMetric", "invalid_metric"),
        ("transformer", "invalid_transformer"),
        ("metric", "invalid_mds_metric"),
        ("missingDataHandling", "invalid_handling"),
        ("pcaType", "invalid_pca"),
        ("solver", "invalid_solver"),
        ("outputFormat", "xml"),
    ],
)
def test_invalid_enum_values_rejected(field, value):
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(router_payload(**{field: value}))
    assert field in exc.value.messages


@pytest.mark.parametrize("field", ["mdsDimensions", "nInit", "maxIter"])
@pytest.mark.parametrize("value", [0, -1])
def test_positive_integers_required(field, value):
    """These are forwarded to scikit-learn, which needs at least one."""
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(router_payload(**{field: value}))
    assert field in exc.value.messages


@pytest.mark.parametrize("field", ["pcaDimensions", "tol", "iteratedPower"])
def test_non_positive_pca_values_allowed(field):
    """A value <= 0 means "choose automatically" for these PCA parameters."""
    result = InputParametersSchema().load(
        router_payload(concatOutput=True, reduceDimensions=True, **{field: -1})
    )
    assert result.reduce_dimensions is True


def test_reduce_dimensions_requires_concat_output():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(
            router_payload(concatOutput=False, reduceDimensions=True)
        )
    assert "reduceDimensions" in exc.value.messages


def test_reduce_dimensions_with_concat_output_is_valid():
    result = InputParametersSchema().load(
        router_payload(concatOutput=True, reduceDimensions=True)
    )
    assert result.concat_output is True
    assert result.reduce_dimensions is True


def test_concat_output_without_reduce_dimensions_is_valid():
    result = InputParametersSchema().load(router_payload(concatOutput=True))
    assert result.concat_output is True
    assert result.reduce_dimensions is False


def test_frontend_validation_accepts_empty_form():
    """The micro frontend renders the initially empty form without errors."""
    assert _frontend_schema(InputParametersSchema).load({}) == {}


def test_frontend_validation_reports_cross_field_error():
    errors = _frontend_schema(InputParametersSchema).load({"reduceDimensions": True})
    assert "reduceDimensions" in errors


def test_all_schema_fields_are_rendered():
    """A field missing from the groups would never show up in the form."""
    rendered = {field for _, fields, _ in INPUT_FIELD_GROUPS for field in fields}
    assert rendered == set(InputParametersSchema().fields)


# --- ROUTING STEP ---


def test_routing_step_accepts_pipeline_fields():
    result = RoutingStepParametersSchema(unknown=EXCLUDE).load(
        {"pipeline_instrumentation": WU_PALMER_PLUGIN, "pipeline_genre": MAPPING_PLUGIN}
    )
    assert result["pipeline_instrumentation"] == WU_PALMER_PLUGIN
    assert result["pipeline_genre"] == MAPPING_PLUGIN


@pytest.mark.parametrize("option", sorted(PIPELINE_OPTIONS))
def test_routing_step_accepts_every_offered_option(option):
    """Every entry of the rendered dropdown has to be loadable."""
    result = RoutingStepParametersSchema(unknown=EXCLUDE).load(
        {"pipeline_genre": option, "pipeline_instrumentation": WU_PALMER_PLUGIN}
    )
    assert result["pipeline_genre"] == option


def test_routing_step_rejects_unknown_field():
    with pytest.raises(ValidationError) as exc:
        RoutingStepParametersSchema(unknown=EXCLUDE).load(
            {"unexpected": WU_PALMER_PLUGIN}
        )
    assert "unexpected" in exc.value.messages


def test_routing_step_rejects_invalid_pipeline():
    with pytest.raises(ValidationError) as exc:
        RoutingStepParametersSchema(unknown=EXCLUDE).load({"pipeline_genre": "Bogus"})
    assert "pipeline_genre" in exc.value.messages


def test_routing_step_rejects_only_none_selections():
    """Without a single pipeline there would be nothing left to compute."""
    with pytest.raises(ValidationError):
        RoutingStepParametersSchema(unknown=EXCLUDE).load(
            {"pipeline_genre": NONE_PLUGIN, "pipeline_instrumentation": NONE_PLUGIN}
        )


def test_routing_step_rejects_empty_selection():
    with pytest.raises(ValidationError):
        RoutingStepParametersSchema(unknown=EXCLUDE).load({"pipeline_genre": ""})


def test_routing_step_accepts_mixed_selections():
    result = RoutingStepParametersSchema(unknown=EXCLUDE).load(
        {"pipeline_genre": NONE_PLUGIN, "pipeline_instrumentation": WU_PALMER_PLUGIN}
    )
    assert result["pipeline_genre"] == NONE_PLUGIN
    assert result["pipeline_instrumentation"] == WU_PALMER_PLUGIN


def test_routing_step_frontend_accepts_empty_form():
    """The micro frontend legitimately validates a form without a selection."""
    assert _frontend_schema(RoutingStepParametersSchema).load({}) == {}


def test_routing_step_frontend_reports_invalid_option():
    errors = _frontend_schema(RoutingStepParametersSchema).load(
        {"pipeline_genre": "Bogus"}
    )
    assert "pipeline_genre" in errors


def test_pipeline_field_prefix_matches_option_keys():
    assert PIPELINE_FIELD_PREFIX == "pipeline_"
    assert set(PIPELINE_OPTIONS) == {
        NONE_PLUGIN,
        WU_PALMER_PLUGIN,
        ONE_HOT_PLUGIN,
        MAPPING_PLUGIN,
    }


# --- ENUM PARITY WITH THE CALLED PLUGINS ---

# The router copies these enums so its form can be rendered without importing
# the other plugins. The payloads are built from the member *names*, so a
# renamed member in the source plugin would break the pipeline at runtime.
ENUM_SOURCES = [
    ("mapping_distances.schemas", "DistanceMetricEnum", DistanceMetricEnum),
    ("transformer.schemas", "TransformersEnum", TransformersEnum),
    ("attribute_mds.schemas", "MetricEnum", MetricEnum),
    ("attribute_mds.schemas", "MissingDataHandling", MissingDataHandling),
    ("pca.schemas", "SolverEnum", SolverEnum),
    ("pca.schemas", "PCATypeEnum", PCATypeEnum),
]


@pytest.mark.parametrize("module_name,enum_name,copied_enum", ENUM_SOURCES)
def test_copied_enums_match_source_plugin(app, module_name, enum_name, copied_enum):
    # The plugin source folders are only importable once the app loaded them.
    source_enum = getattr(import_module(module_name), enum_name)
    assert {member.name for member in copied_enum} == {
        member.name for member in source_enum
    }


def test_pipeline_plugins_are_installed(app):
    """The routing step builds urls from these names, so they have to exist."""
    from qhana_plugin_runner.util.plugins import QHAnaPluginBase

    installed = {plugin.name for plugin in QHAnaPluginBase.get_plugins().values()}
    assert set(PIPELINE_PLUGINS.values()) <= installed
