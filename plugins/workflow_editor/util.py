from io import BytesIO
from xml.etree import ElementTree


def extract_wf_properties(bpmn: str):
    id_ = "unknown"
    name = id_
    version = "0"
    for _event, node in ElementTree.iterparse(
        BytesIO(bpmn.encode(encoding="utf-8")), ["start"]
    ):
        if node.tag == "{http://www.omg.org/spec/BPMN/20100524/MODEL}definitions":
            continue
        if (
            node.tag == "{http://www.omg.org/spec/BPMN/20100524/MODEL}process"
            or node.tag.endswith("process")
        ):
            id_ = node.attrib["id"]
            name = node.attrib.get("name", id_)
            version = node.attrib.get(
                "{http://camunda.org/schema/1.0/bpmn}versionTag", version
            )
        break
    return id_, name, version
