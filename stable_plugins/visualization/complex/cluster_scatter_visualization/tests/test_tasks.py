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

import json

import pytest

from tests.utils import MockResponse

from cluster_scatter_visualization.tasks import _get_plot

ENTITY_URL = "http://example.com/concatenated.json"
MAPPING_URL = "http://example.com/concatenated_dimension_mapping.json"

ENTITIES_3D = [
    {"ID": "e1", "href": "h1", "dim0": 1, "dim1": 2, "dim2": 3},
    {"ID": "e2", "href": "h2", "dim0": 4, "dim1": 5, "dim2": 6},
]

MAPPING_3D = [
    {
        "ID": "dim0",
        "href": "",
        "inputIndex": 0,
        "source": "color",
        "sourceUrl": "http://example.com/color.csv",
        "zipMember": "",
        "sourceDimension": "dim0",
    },
    {
        "ID": "dim1",
        "href": "",
        "inputIndex": 0,
        "source": "color",
        "sourceUrl": "http://example.com/color.csv",
        "zipMember": "",
        "sourceDimension": "dim1",
    },
    {
        "ID": "dim2",
        "href": "",
        "inputIndex": 1,
        "source": "shape",
        "sourceUrl": "http://example.com/shape.csv",
        "zipMember": "",
        "sourceDimension": "dim0",
    },
]


@pytest.fixture
def responses(monkeypatch):
    """Serve entity and mapping files from memory to both ``open_url`` users."""
    served = {}

    def mock_open_url(url, *args, **kwargs):
        return served[url]

    monkeypatch.setattr("cluster_scatter_visualization.tasks.open_url", mock_open_url)
    monkeypatch.setattr(
        "qhana_plugin_runner.plugin_utils.dimension_mapping.open_url", mock_open_url
    )
    return served


def _serve(responses, url, entities):
    responses[url] = MockResponse(url, "application/json", json_data=entities)


def _axis_titles(html: str) -> dict:
    """Extract the axis titles from the plotly figure embedded in ``html``.

    The rendered page also contains the plotly bundle, so this helper reads the
    figure from the arguments of the final ``Plotly.newPlot`` call.
    """
    decoder = json.JSONDecoder()
    call = html[html.rindex("Plotly.newPlot(") :]
    _div_id, position = decoder.raw_decode(call, call.index('"'))
    _data, position = decoder.raw_decode(call, call.index("[", position))
    layout, _position = decoder.raw_decode(call, call.index("{", position))

    # 3d plots keep their axes in a scene.
    axes = layout.get("scene", layout)
    return {
        axis: axes.get(f"{axis}axis", {}).get("title", {}).get("text")
        for axis in ("x", "y", "z")
    }


def test_axes_keep_their_default_titles_without_a_mapping(responses):
    _serve(responses, ENTITY_URL, ENTITIES_3D)

    html, _name = _get_plot(ENTITY_URL, None, None, None, full_html=False)

    assert _axis_titles(html) == {"x": "x", "y": "y", "z": "z"}


def test_axes_are_labelled_with_the_feature_names(responses):
    _serve(responses, ENTITY_URL, ENTITIES_3D)
    _serve(responses, MAPPING_URL, MAPPING_3D)

    html, _name = _get_plot(ENTITY_URL, None, None, MAPPING_URL, full_html=False)

    assert _axis_titles(html) == {
        "x": "color (dim0)",
        "y": "color (dim1)",
        "z": "shape",
    }


def test_two_dimensional_plots_are_labelled(responses):
    entities = [
        {"ID": e["ID"], "href": e["href"], **{"dim0": e["dim0"], "dim1": e["dim1"]}}
        for e in ENTITIES_3D
    ]
    _serve(responses, ENTITY_URL, entities)
    _serve(responses, MAPPING_URL, MAPPING_3D[:2])

    html, _name = _get_plot(ENTITY_URL, None, None, MAPPING_URL, full_html=False)

    titles = _axis_titles(html)
    assert titles["x"] == "color (dim0)"
    assert titles["y"] == "color (dim1)"


def test_constant_axis_of_one_dimensional_plots_keeps_its_default_title(responses):
    _serve(responses, ENTITY_URL, [{"ID": "e1", "href": "h1", "dim0": 1}])
    _serve(responses, MAPPING_URL, MAPPING_3D[:1])

    html, _name = _get_plot(ENTITY_URL, None, None, MAPPING_URL, full_html=False)

    titles = _axis_titles(html)
    assert titles["x"] == "color"
    assert titles["y"] == "y"


def test_dimensions_without_a_mapping_entry_keep_their_default_titles(responses):
    _serve(responses, ENTITY_URL, ENTITIES_3D)
    _serve(responses, MAPPING_URL, MAPPING_3D[:1])

    html, _name = _get_plot(ENTITY_URL, None, None, MAPPING_URL, full_html=False)

    assert _axis_titles(html) == {"x": "color", "y": "y", "z": "z"}


def test_labels_follow_the_dimension_names_past_the_ninth_dimension(responses):
    """``ensure_array`` orders values lexicographically, so ``dim10`` precedes ``dim2``."""
    entity = {"ID": "e1", "href": "h1"}
    entity.update({f"dim{i}": i for i in range(12)})
    _serve(responses, ENTITY_URL, [entity])
    _serve(
        responses,
        MAPPING_URL,
        [
            {
                "ID": f"dim{i}",
                "href": "",
                "inputIndex": i,
                "source": f"feature{i}",
                "sourceUrl": f"http://example.com/feature{i}.csv",
                "zipMember": "",
                "sourceDimension": "dim0",
            }
            for i in range(12)
        ],
    )

    html, _name = _get_plot(ENTITY_URL, None, None, MAPPING_URL, full_html=False)

    assert _axis_titles(html) == {
        "x": "feature0",
        "y": "feature1",
        "z": "feature10",
    }
