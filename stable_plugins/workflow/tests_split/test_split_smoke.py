from pathlib import Path
import pytest
import xml.etree.ElementTree as ET

from splitting.split import split_workflow, SplitNotSupported

ROOT = Path(__file__).resolve().parents[0] 
WF_EDITOR = ROOT.parent / "workflow_editor"
BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
NS = {
    "bpmn": BPMN_NS,
    "bpmndi": BPMNDI_NS,
}

out_dir = ROOT / "_out"
out_dir.mkdir(exist_ok=True)


def _read_bpmn(name: str) -> str:
    return (WF_EDITOR / "tests" / "bpmn" / name).read_text(encoding="utf-8")


def _write_outputs(case_name: str, exec_xml: str, rest_xml: str):
    stem = Path(case_name).stem
    (out_dir / f"{stem}_exec.bpmn").write_text(exec_xml, encoding="utf-8")
    (out_dir / f"{stem}_rest.bpmn").write_text(rest_xml, encoding="utf-8")


def _assert_well_formed(xml_str: str):
    ET.fromstring(xml_str)


def _parse(xml_str: str) -> ET.Element:
    return ET.fromstring(xml_str)


def _assert_xor_semantics(rest_xml: str):
    root = _parse(rest_xml)
    proc = root.find(".//bpmn:process", NS)
    assert proc is not None

    gw = proc.find(".//bpmn:exclusiveGateway", NS)
    assert gw is not None

    cond = proc.find(".//bpmn:sequenceFlow/bpmn:conditionExpression", NS)
    assert cond is not None

    default_id = gw.get("default")
    assert default_id is not None and default_id.strip() != ""
    default_flow = proc.find(f".//bpmn:sequenceFlow[@id='{default_id}']", NS)
    assert default_flow is not None


def _assert_participant_and_lanes(xml_str: str):
    root = _parse(xml_str)
    assert root.find(".//bpmn:participant", NS) is not None
    assert root.find(".//bpmn:laneSet", NS) is not None


def _assert_di_present(xml_str: str):
    root = _parse(xml_str)
    assert root.find(".//bpmndi:BPMNDiagram", NS) is not None


def _all_lane_refs_exist(xml_str: str):
    root = _parse(xml_str)
    ids = {e.get("id") for e in root.findall(".//*[@id]")}
    for ref in root.findall(".//bpmn:lane//bpmn:flowNodeRef", NS):
        assert (ref.text or "").strip() in ids, f"Dangling lane flowNodeRef: {(ref.text or '').strip()}"


def _all_association_refs_exist(xml_str: str):
    root = _parse(xml_str)
    ids = {e.get("id") for e in root.findall(".//*[@id]")}
    for assoc in root.findall(".//bpmn:association", NS):
        src = assoc.get("sourceRef")
        tgt = assoc.get("targetRef")
        assert src in ids, f"Dangling association sourceRef: {src}"
        assert tgt in ids, f"Dangling association targetRef: {tgt}"


def _all_di_refs_exist(xml_str: str):
    root = _parse(xml_str)
    ids = {e.get("id") for e in root.findall(".//*[@id]")}
    proc = root.find(".//bpmn:process", NS)
    if proc is not None and proc.get("id"):
        ids.add(proc.get("id"))
    for shape in root.findall(".//bpmndi:*[@bpmnElement]", NS):
        ref = shape.get("bpmnElement")
        assert ref in ids, f"Dangling DI bpmnElement ref: {ref}"


def test_split_smoke_tc01_roundtrip():
    case = "tc01_hello_world_adhoc.bpmn"
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    assert "_exec" in exec_xml
    assert "_rest" in rest_xml

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    assert "adHocSubProcess" in rest_xml
    assert "adHocSubProcess" not in exec_xml
    assert ("UserTask_Input" in rest_xml) or ("Input (UserTask)" in rest_xml) or ("userTask" in rest_xml)

    _write_outputs(case, exec_xml, rest_xml)


@pytest.mark.parametrize("case", [
    "tc02_xor_true.bpmn",
    "tc03_xor_default.bpmn",
])
def test_split_smoke_xor_gateways(case):
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    assert "_exec" in exec_xml
    assert "_rest" in rest_xml

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    assert "exclusiveGateway" in exec_xml
    assert "exclusiveGateway" in rest_xml

    _assert_xor_semantics(rest_xml)

    assert "qHAnaServiceTask" in exec_xml or "qhanaPlugin" in exec_xml or ('camunda:topic="qhana-task"' in exec_xml)
    assert ("UserTask_Input" in rest_xml) or ("Input (UserTask)" in rest_xml) or ("userTask" in rest_xml)

    _write_outputs(case, exec_xml, rest_xml)


@pytest.mark.parametrize("case", [
    "tc04_and_split.bpmn",
    "tc05_and_join.bpmn",
])
def test_split_smoke_and_gateways(case):
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    assert "_exec" in exec_xml
    assert "_rest" in rest_xml

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    assert "parallelGateway" in exec_xml
    assert "parallelGateway" in rest_xml

    assert "qHAnaServiceTask" in exec_xml or "qhanaPlugin" in exec_xml or ('camunda:topic="qhana-task"' in exec_xml)
    assert ("UserTask_Input" in rest_xml) or ("Input (UserTask)" in rest_xml) or ("userTask" in rest_xml)

    _write_outputs(case, exec_xml, rest_xml)


def test_split_smoke_tc06_task_types():
    case = "tc06_task_types.bpmn"
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    for tag in ["manualTask", "scriptTask", "businessRuleTask", "sendTask", "receiveTask"]:
        assert tag in rest_xml, f"Expected {tag} in REST split"

    assert "qHAnaServiceTask" in exec_xml or ('camunda:topic="qhana-task"' in exec_xml)

    _write_outputs(case, exec_xml, rest_xml)


def test_split_smoke_tc07_artifacts_and_associations():
    case = "tc07_artifacts_associations.bpmn"
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    assert "dataObjectReference" in rest_xml
    assert "textAnnotation" in rest_xml
    assert "association" in rest_xml

    assert "qhana:customFlag" in exec_xml

    _all_association_refs_exist(exec_xml)
    _all_association_refs_exist(rest_xml)

    _write_outputs(case, exec_xml, rest_xml)


def test_split_smoke_tc08_pool_lane():
    case = "tc08_pool_lane.bpmn"
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    _assert_participant_and_lanes(exec_xml)
    _assert_participant_and_lanes(rest_xml)

    _all_lane_refs_exist(exec_xml)
    _all_lane_refs_exist(rest_xml)

    _write_outputs(case, exec_xml, rest_xml)


def test_split_smoke_tc09_di_and_extensions():
    case = "tc09_di_extensions.bpmn"
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    assert 'camunda:asyncBefore="true"' in exec_xml
    assert 'qhana:customAttr="abc"' in exec_xml
    assert "extensionElements" in exec_xml

    _assert_di_present(exec_xml)
    _assert_di_present(rest_xml)
    _all_di_refs_exist(exec_xml)
    _all_di_refs_exist(rest_xml)

    _write_outputs(case, exec_xml, rest_xml)


def test_split_smoke_tc10_service_task():
    case = "tc10_service_task.bpmn"
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    assert 'camunda:topic="qhana-task"' in exec_xml or "qHAnaServiceTask" in exec_xml
    assert 'camunda:topic="other-topic"' in rest_xml

    assert 'camunda:topic="other-topic"' not in exec_xml

    _write_outputs(case, exec_xml, rest_xml)


def test_split_smoke_tc11_message_attrs():
    case = "tc11_message_attrs.bpmn"
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    assert "sendTask" in rest_xml
    assert "receiveTask" in rest_xml

    root = _parse(rest_xml)
    send_task = root.find(".//bpmn:sendTask", NS)
    recv_task = root.find(".//bpmn:receiveTask", NS)
    assert send_task is not None
    assert recv_task is not None

    assert send_task.get("messageRef")
    assert send_task.get("operationRef")
    assert recv_task.get("messageRef")
    assert recv_task.get("operationRef") is not None

    if recv_task.get("instantiate") is not None:
        assert recv_task.get("instantiate") in {"true", "false"}

    assert "qHAnaServiceTask" in exec_xml or 'camunda:topic="qhana-task"' in exec_xml

    _write_outputs(case, exec_xml, rest_xml)


def test_split_smoke_tc12_lane_associated_cleanup():
    case = "tc12_lane_associated_cleanup.bpmn"
    bpmn = _read_bpmn(case)
    exec_xml, rest_xml = split_workflow(bpmn)

    _assert_well_formed(exec_xml)
    _assert_well_formed(rest_xml)

    _assert_participant_and_lanes(exec_xml)
    _assert_participant_and_lanes(rest_xml)
    _assert_di_present(exec_xml)
    _assert_di_present(rest_xml)

    _all_lane_refs_exist(exec_xml)
    _all_lane_refs_exist(rest_xml)
    _all_association_refs_exist(exec_xml)
    _all_association_refs_exist(rest_xml)
    _all_di_refs_exist(exec_xml)
    _all_di_refs_exist(rest_xml)

    assert "userTask" in rest_xml or "manualTask" in rest_xml
    assert "qHAnaServiceTask" in exec_xml or 'camunda:topic="qhana-task"' in exec_xml

    _write_outputs(case, exec_xml, rest_xml)