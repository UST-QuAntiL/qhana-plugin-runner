import re

import dash_cytoscape as cyto
from dash import Dash, Input, Output, callback, ctx, dcc, html

"""
Requires pandas, dash and dash-cytoscape installed.

BLUE lines represent an input and ORANGE an output.

Run this with the following command from the repository root:
> poetry run python docs/plugin-connection-graph/graph.py

Open http://localhost:8050 to view the diagram.
"""


def normalize_mimetype_like(mimetype: str):
    if not mimetype or mimetype == "*":
        return "*/*"

    if "/" not in mimetype:
        return mimetype + "/*"

    first, second = mimetype.split("/", maxsplit=1)
    if not first:
        first = "*"
    if not second:
        second = "*"

    return f"{first}/{second}"


def prepare_data_metadata(metadata):
    data_type = normalize_mimetype_like(metadata["dataType"])
    content_types = [normalize_mimetype_like(c) for c in metadata["contentType"]]
    return (data_type, content_types, metadata.get("required", False))


def get_plugin_info(plugin, metadata):
    info = {}
    info["identifier"] = plugin.identifier
    info["id"] = metadata["name"]
    info["name"] = metadata["title"]
    info["input"] = [
        prepare_data_metadata(m) for m in metadata["entryPoint"].get("dataInput", [])
    ]
    info["output"] = [
        prepare_data_metadata(m) for m in metadata["entryPoint"].get("dataOutput", [])
    ]
    return info


def get_plugins():
    from qhana_plugin_runner import create_app
    from qhana_plugin_runner.util.plugins import QHAnaPluginBase

    app = create_app(silent_log=True)

    plugins = [p for p in QHAnaPluginBase.get_plugins().values()]

    with app.test_client() as c:
        plugin_metadata = [c.get(f"/plugins/{p.identifier}/").json for p in plugins]

    return {
        m["name"]: get_plugin_info(p, m)
        for (p, m) in zip(plugins, plugin_metadata)
        if "status" not in m and "code" not in m
    }


def break_camel_case(text):
    """Break CamelCase text into space-separated words"""
    return re.sub(r"(?<!^)(?<![A-Z])(?<![^a-zA-Z])(?=[A-Z])", " ", text)


def break_hyphen(text):
    """Break hyphenated words longer than 16 characters by adding space after hyphen"""
    words = text.split(" ")
    result = []
    for word in words:
        if len(word) > 16 and "-" in word:
            word = word.replace("-", "- ")
        result.append(word)
    return " ".join(result)


def break_word(text):
    """Breaks a word by CamelCase or hyphens"""
    return break_hyphen(break_camel_case(text))


def get_unique_data_content_type(plugins):
    data_type_set = set()
    content_type_set = set()
    for _plugin_name, plugin in plugins.items():
        joined_ary = plugin["input"] + plugin["output"]

        for entry in joined_ary:
            data_type_set.add(entry[0])
            for inner_entry in entry[1]:
                content_type_set.add(inner_entry)
    return data_type_set, content_type_set


def create_graph(plugins):
    """
    Creates the nodes and edges required for a dash graph

    Input:
        {
            'First Plugin (@v1.0.0)': {
                'input': [
                    [...],
                    [
                        'example-data-type',
                        ['example-content-type'],
                        True (depending on whether this input is required)
                    ]
                ],
                'output': [
                    [...],
                    [
                        'example-data-type',
                        ['example-content-type'],
                        False (depending on whether this output is required)
                    ]
                ]
            },
            'Second Plugin (@v2.0.0)': ...
        }
    Output:
        [
            {
                'data': {
                    'id': 'example-data-type',
                    'label': 'example- data- type',
                },
                'classes': 'data-types'
            },
            {
                'data': {
                    'id': 'First Plugin (@v1.0.0) input - example-data-type',
                    'label': 'First Plugin (@v1.0.0)'
                }
            },
            {
                'data': {
                    'id': 'First Plugin (@v1.0.0) output - example-data-type',
                    'label': 'First Plugin (@v1.0.0)'
                }
            },
            {
                'data': {
                    'id': 'Second Plugin (@v2.0.0) input - example-data-type',
                    'label': 'Second Plugin (@v2.0.0)'
                }
            },
            ...
        ]
    """
    data_types = list(get_unique_data_content_type(plugins)[0])

    # Nodes:
    elements = []
    for data_type in data_types:
        elements.append(
            {
                "data": {"id": data_type, "label": break_word(data_type)},
                "classes": "data-types",
            }
        )
        for (
            plugin_name,
            plugin,
        ) in (
            plugins.items()
        ):  # example plugin = {'input': [['executable/circuit', ['text/x-qasm'], True]], 'output': [['circuit/*', ['text/html'], True]]}
            inputs = []
            for input in plugin["input"]:
                inputs.append(input[0])
            outputs = []
            for output in plugin["output"]:
                outputs.append(output[0])

            if inputs.__contains__(data_type):
                elements.append(
                    {
                        "data": {
                            "id": plugin_name + " input - " + data_type,
                            "label": break_word(plugin_name.split(" (@")[0]),
                        }
                    }
                )
            if outputs.__contains__(data_type):
                elements.append(
                    {
                        "data": {
                            "id": plugin_name + " output - " + data_type,
                            "label": break_word(plugin_name.split(" (@")[0]),
                        }
                    }
                )

    # Edges:
    for data_type in data_types:
        for plugin_name, plugin in plugins.items():
            for input in plugin["input"]:
                if data_type == input[0]:
                    elements.append(
                        {
                            "data": {
                                "source": plugin_name + " input - " + data_type,
                                "target": input[0],
                            },
                            "classes": "input-edge",
                        }
                    )

            for output in plugin["output"]:
                if data_type == output[0]:
                    elements.append(
                        {
                            "data": {
                                "source": output[0],
                                "target": plugin_name + " output - " + data_type,
                            },
                            "classes": "output-edge",
                        }
                    )

    return elements


def show_graph(data):
    cyto.load_extra_layouts()

    app = Dash()

    stylesheet = [
        {
            "selector": "node",
            "style": {
                "background-color": "#d9d9d9",
                "width": "80px",
                "height": "80px",
                "font-size": "10px",
                "text-wrap": "wrap",
                "text-max-width": "50px",
                "content": "data(label)",
                "text-halign": "center",
                "text-valign": "center",
                "overflow-wrap": "anywhere",
                "word-break": "break-all",
                "shape": "square",
            },
        },
        {
            "selector": ".data-types",
            "style": {
                "background-color": "#a9a9a9",
                "width": "150px",
                "height": "80px",
                "font-size": "14px",
                "shape": "square",
            },
        },
        {
            "selector": ".output-edge",
            "style": {
                "line-color": "cyan",
                "mid-source-arrow-color": "cyan",
                "mid-source-arrow-shape": "triangle",
                "arrow-scale": 2.0,
            },
        },
        {
            "selector": ".input-edge",
            "style": {
                "line-color": "orange",
                "mid-source-arrow-color": "orange",
                "mid-source-arrow-shape": "triangle",
                "arrow-scale": 2.0,
            },
        },
    ]

    app.layout = html.Div(
        [
            cyto.Cytoscape(
                id="plugin-graph",
                style={
                    # 'width': '2560px',
                    # 'height': '1440px'
                    "width": "1000px",
                    "height": "1000px",
                },
                stylesheet=stylesheet,
                elements=data,
                wheelSensitivity=0.1,
                layout={
                    "name": "klay",
                    "klay": {
                        "aspectRatio": 0.5,
                        "nodePlacement": "LINEAR_SEGMENTS",
                        # 'direction': 'LEFT',
                        "direction": "UP",
                        "spacing": 40,
                        # 'inLayerSpacingFactor': 5.0
                    },
                },
            ),
            html.Div(
                children=[
                    html.H4("Legend"),
                    html.Div(
                        children=[
                            html.Span(
                                style={
                                    "background-color": "orange",
                                    "padding": "0px 5px",
                                    "margin-right": "5px",
                                }
                            ),
                            "Used as Input",
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Span(
                                style={
                                    "background-color": "cyan",
                                    "padding": "0px 5px",
                                    "margin-right": "5px",
                                }
                            ),
                            "Output produced",
                        ]
                    ),
                ],
                style={
                    "position": "absolute",
                    "top": "10px",
                    "right": "10px",
                    "border": "1px solid black",
                    "padding": "10px",
                    "background-color": "white",
                },
            ),
            html.Button("Download graph", id="svg"),
        ]
    )

    @callback(Output("plugin-graph", "generateImage"), Input("svg", "n_clicks"))
    def get_image(get_svg_clicks):
        if ctx.triggered:
            return {"type": "svg", "action": "download"}
        return {}

    app.run(debug=True)


if __name__ == "__main__":
    plugins = get_plugins()
    data = create_graph(plugins)
    show_graph(data)
