import re
from inspect import getfile
from pathlib import Path

ALLOWED_TYPES_DOC = "data-formats/allowed-types.md"


def type_anchor(value: str, prefix: str) -> str:
    """Build the documentation anchor for a data_type or content_type.

    The anchors are the link targets in ``docs/data-formats/allowed-types.md``.
    Use the prefix ``dt`` for data types and ``ct`` for content types.

    ``tests/allowed_types.py`` contains a copy of this function (this script must run
    standalone during the doc build and cannot import from the test package).
    ``tests/test_allowed_types_docs.py`` asserts that both agree.
    """
    # "*" would slugify to nothing, so it is spelled out ("*" -> "wildcard",
    # "entity/*" -> "entity-wildcard")
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower().replace("*", "wildcard")).strip("-")
    return f"{prefix}-{slug}"


def link_types(types, prefix: str) -> str:
    """Render types as links into the allowed types documentation."""
    # "#target" links resolve against the explicit MyST targets of the whole project, unlike
    # "path.md#target" links, which MyST can only resolve for auto-generated heading anchors.
    return ", ".join(f"[`{t}`](#{type_anchor(t, prefix)})" for t in sorted(types))


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
    metadata["dataType"] = normalize_mimetype_like(metadata["dataType"])
    metadata["contentType"] = [
        normalize_mimetype_like(c) for c in metadata["contentType"]
    ]
    return metadata


def collect_raw_types(data_metadata):
    """Collect the declared (not normalized) data and content types.

    ``prepare_data_metadata`` normalizes the metadata in place, so the declared values have to
    be captured before it runs. The overview lists them verbatim to match the allowed types
    documentation.
    """
    data_types = set()
    content_types = set()
    for m in data_metadata:
        data_types.add(m["dataType"])
        content_types.update(m["contentType"])
    return data_types, content_types


def get_plugin_info(plugin, metadata, base_path):
    info = {}
    info["identifier"] = plugin.identifier
    info["id"] = metadata["name"]
    info["type"] = metadata["type"]
    info["version"] = metadata["version"]
    info["name"] = metadata["title"]
    info["description"] = metadata["description"] if metadata["description"] else ""
    info["tags"] = sorted(metadata["tags"])
    data_input = metadata["entryPoint"].get("dataInput", [])
    data_output = metadata["entryPoint"].get("dataOutput", [])
    info["input_raw_types"], info["input_raw_formats"] = collect_raw_types(data_input)
    info["output_raw_types"], info["output_raw_formats"] = collect_raw_types(data_output)
    info["input"] = [prepare_data_metadata(m) for m in data_input]
    info["output"] = [prepare_data_metadata(m) for m in data_output]
    info["path"] = Path(getfile(type(plugin))).relative_to(base_path)
    return info


def get_plugins(base_path):
    from qhana_plugin_runner import create_app
    from qhana_plugin_runner.util.plugins import QHAnaPluginBase

    app = create_app(silent_log=True)

    plugins = [p for p in QHAnaPluginBase.get_plugins().values()]

    with app.test_client() as c:
        plugin_metadata = [c.get(f"/plugins/{p.identifier}/").json for p in plugins]

    return [
        get_plugin_info(p, m, base_path)
        for (p, m) in zip(plugins, plugin_metadata)
        if "status" not in m and "code" not in m
    ]


def write_index(doc, plugins):
    doc.write(":::{list-table} Plugin Overview\n")
    doc.write(":header-rows: 1\n")
    doc.write(":width: 100%\n")
    doc.write(":widths: 30 10 30\n\n")
    doc.write("* - Plugin\n  - Type\n  - Tags\n")
    sep = "\n\n    "
    for p in sorted(plugins, key=lambda p: p["name"]):
        doc.write(f"* - [{p['name']} (@{p['version']})](#{p['id']})\n\n")
        doc.write(f"    {p['identifier']}\n")
        doc.write(f"  - {p['type']}\n")
        doc.write(f"  - {sep.join(p['tags'])}\n")
    doc.write("\n:::\n\n")


def write_merged_data(doc, plugins):
    tags = set()
    input_formats = set()
    output_formats = set()
    input_datatypes = set()
    output_datatypes = set()

    for p in plugins:
        tags.update(p["tags"])
        input_formats.update(p["input_raw_formats"])
        input_datatypes.update(p["input_raw_types"])
        output_formats.update(p["output_raw_formats"])
        output_datatypes.update(p["output_raw_types"])

    doc.write("## Overview\n\n")
    doc.write(f"**Used tags:** {', '.join(f'`{t}`' for t in sorted(tags))}\n\n")
    doc.write(
        f"Every type below links to its entry in the list of "
        f"[allowed data types and content types]({ALLOWED_TYPES_DOC}).\n\n"
    )
    doc.write(f"**Input formats:** {link_types(input_formats, 'ct')}\\\n")
    doc.write(f"**Output formats:** {link_types(output_formats, 'ct')}\n\n")
    doc.write(f"**Input datatypes:** {link_types(input_datatypes, 'dt')}\\\n")
    doc.write(f"**Output datatypes:** {link_types(output_datatypes, 'dt')}\n\n")


def write_plugin(doc, p):
    doc.write(f"({p['id']})=\n")
    doc.write(f"### {p['name']} (@{p['version']})\n\n")
    doc.write(f"{p['type']} – {', '.join(p['tags'])}\\\n")
    doc.write(f"*Path:* {{file}}`{p['path']}`\n\n")
    doc.write(p["description"])
    doc.write("\n\n")
    if p["input"]:
        doc.write("**Inputs:**\n\n")
        doc.write("| Data Type | Content Type | Required |\n")
        doc.write("|-----------|--------------| :------: |\n")
        for data_input in p["input"]:
            doc.write(
                f"|{data_input['dataType']}|{', '.join(data_input['contentType'])}|{'✓' if data_input['required'] else '╳'}|\n"
            )
        doc.write("\n\n")
    if p["output"]:
        doc.write("**Outputs:**\n\n")
        doc.write("| Data Type | Content Type | Always |\n")
        doc.write("|-----------|--------------| :----: |\n")
        for data_output in p["output"]:
            doc.write(
                f"|{data_output['dataType']}|{', '.join(data_output['contentType'])}|{'✓' if data_output['required'] else '╳'}|\n"
            )
        doc.write("\n\n")


def create_plugin_doc():
    import warnings

    doc_path = Path(".").resolve()
    if doc_path.name != "docs":
        doc_path = doc_path / "docs"

    base_path = doc_path.parent

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", module="apispec.ext.marshmallow.common", lineno=139
        )
        plugins = get_plugins(base_path)

    doc_path = doc_path / "all-plugins.md"

    with doc_path.open("wt") as doc:
        doc.write("# All Plugins\n\n")
        write_index(doc, plugins)
        write_merged_data(doc, plugins)
        doc.write("## Plugins\n\n")
        for plugin in sorted(plugins, key=lambda p: p["name"]):
            write_plugin(doc, plugin)
        doc.write("\n")


if __name__ == "__main__":
    from os import environ

    if not environ.get("PLUGIN_FOLDERS"):
        # TODO remove after testing?
        environ["PLUGIN_FOLDERS"] = (
            "./plugins:./stable_plugins/classical_ml/data_preparation:./stable_plugins/classical_ml/scikit_ml:./stable_plugins/data_synthesis:./stable_plugins/demo:./stable_plugins/file_utils:./stable_plugins/infrastructure:./stable_plugins/muse:./stable_plugins/nisq_analyzer:./stable_plugins/quantum_ml/max_cut:./stable_plugins/quantum_ml/pennylane_qiskit_ml:./stable_plugins/quantum_ml/qiskit_ml:./stable_plugins/visualization/complex:./stable_plugins/visualization/file_types:./stable_plugins/workflow"
        )

    create_plugin_doc()
