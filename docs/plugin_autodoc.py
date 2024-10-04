from inspect import getfile
from pathlib import Path


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


def get_plugin_info(plugin, metadata, base_path):
    info = {}
    info["identifier"] = plugin.identifier
    info["id"] = metadata["name"]
    info["type"] = metadata["type"]
    info["version"] = metadata["version"]
    info["name"] = metadata["title"]
    info["description"] = metadata["description"] if metadata["description"] else ""
    info["tags"] = sorted(metadata["tags"])
    info["input"] = [
        prepare_data_metadata(m) for m in metadata["entryPoint"].get("dataInput", [])
    ]
    info["output"] = [
        prepare_data_metadata(m) for m in metadata["entryPoint"].get("dataOutput", [])
    ]
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
    doc.write(":::{Table} Plugin Overview\n")
    doc.write(":width: 100%\n")
    doc.write(":widths: 30 20 10 50\n\n")
    doc.write("| Plugin | ID | Type | Tags |\n")
    doc.write("|--------|----|------|------|\n")
    for p in sorted(plugins, key=lambda p: p["name"]):
        doc.write(
            f"|[{p['name']} (@{p['version']})](#{p['id']})|{p['identifier']}|{p['type']}|{', '.join(p['tags'])}|\n"
        )
    doc.write("\n:::\n\n")


def write_merged_data(doc, plugins):
    tags = set()
    input_formats = set()
    output_formats = set()
    input_datatypes = set()
    output_datatypes = set()

    for p in plugins:
        tags.update(p["tags"])
        for input_def in p["input"]:
            input_formats.update(input_def["contentType"])
            input_datatypes.add(input_def["dataType"])
        for output_def in p["output"]:
            output_formats.update(output_def["contentType"])
            output_datatypes.add(output_def["dataType"])

    doc.write("## Overview\n\n")
    doc.write(f"**Used tags:** {', '.join(f'`{t}`' for t in sorted(tags))}\\\n")
    doc.write(
        f"**Input formats:** {', '.join(f'`{f}`' for f in sorted(input_formats))}\\\n"
    )
    doc.write(
        f"**Output formats:** {', '.join(f'`{f}`' for f in sorted(output_formats))}\\\n"
    )
    doc.write(
        f"**Input datatypes:** {', '.join(f'`{f}`' for f in sorted(input_datatypes))}\\\n"
    )
    doc.write(
        f"**Output datatypes:** {', '.join(f'`{f}`' for f in sorted(output_datatypes))}\n\n"
    )


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
                f"|{data_input['dataType']}|{', '.join(data_input['contentType'])}|{'✓' if data_input['required'] else '🗙'}|\n"
            )
        doc.write("\n\n")
    if p["output"]:
        doc.write("**Outputs:**\n\n")
        doc.write("| Data Type | Content Type | Always |\n")
        doc.write("|-----------|--------------| :----: |\n")
        for data_output in p["output"]:
            doc.write(
                f"|{data_output['dataType']}|{', '.join(data_output['contentType'])}|{'✓' if data_output['required'] else '🗙'}|\n"
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
