from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from splitting.split import split_workflow, SplitNotSupported, write_split_outputs
from expected_fixtures import EXPECTED

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
QHANA_NS = "https://github.com/qhana"
NS = {"bpmn": BPMN_NS, "qhana": QHANA_NS}

ROOT = Path(__file__).parent
BPMN_DIR = ROOT.parent / "workflow_editor" / "tests" / "bpmn"
OUT_DIR = ROOT / "_out"


def _ln(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _fingerprint_process(xml_str: str) -> Tuple[List[Tuple[str, str]], List[Tuple[Any, ...]]]:
    """Return (nodes, flows) where nodes is a list of (kind, id) tuples in
    document order and flows is a list of (id, src, tgt, cond_or_None) tuples.

    `kind` distinguishes wrapper ad-hocs from original ones via the
    qhana:fragmentRef attribute, mirroring the fixture format.
    """
    root = ET.fromstring(xml_str)
    proc = root.find(f"{{{BPMN_NS}}}process")
    if proc is None:
        proc = root.find(".//bpmn:process", NS)
    assert proc is not None, "no process element found"

    nodes: List[Tuple[str, str]] = []
    flows: List[Tuple[Any, ...]] = []
    for child in proc:
        local = _ln(child.tag)
        if local == "sequenceFlow":
            cond_elem = child.find("bpmn:conditionExpression", NS)
            cond = (cond_elem.text or "").strip() if cond_elem is not None else None
            flows.append((child.get("id"), child.get("sourceRef"), child.get("targetRef"), cond))
        elif local in ("laneSet", "association"):
            continue
        else:
            cid = child.get("id")
            if cid is None:
                continue
            kind = local
            if local == "adHocSubProcess":
                fref = child.get(f"{{{QHANA_NS}}}fragmentRef")
                if fref:
                    kind = f"adHocSubProcess[wrapper={fref}]"
                else:
                    kind = "adHocSubProcess[original]"
            nodes.append((kind, cid))
    return nodes, flows


def _fragment_task_ids(xml_str: str) -> List[str]:
    """Task ids inside a fragment process, in document order, excluding the
    fragment's own start/end events."""
    root = ET.fromstring(xml_str)
    proc = root.find(f"{{{BPMN_NS}}}process")
    assert proc is not None
    out: List[str] = []
    for child in proc:
        local = _ln(child.tag)
        if local in ("sequenceFlow", "startEvent", "endEvent", "laneSet", "association"):
            continue
        cid = child.get("id")
        if cid:
            out.append(cid)
    return out


def _fragment_flow_ids(xml_str: str) -> List[str]:
    root = ET.fromstring(xml_str)
    proc = root.find(f"{{{BPMN_NS}}}process")
    assert proc is not None
    return [c.get("id") for c in proc if _ln(c.tag) == "sequenceFlow"]

CASES = sorted(EXPECTED.keys())


@pytest.fixture(autouse=True)
def _ensure_out_dir():
    OUT_DIR.mkdir(exist_ok=True)


@pytest.mark.parametrize("case_name", CASES)
def test_case(case_name: str):
    expected = EXPECTED[case_name]
    bpmn_xml = (BPMN_DIR / case_name).read_text(encoding="utf-8")

    if "nsup" in expected:
        with pytest.raises(SplitNotSupported) as excinfo:
            split_workflow(bpmn_xml)
        assert excinfo.value is not None
        return

    stem = Path(case_name).stem
    paths = write_split_outputs(bpmn_xml, OUT_DIR, stem)
    assert len(paths) >= 1, "at least the main file must be written"

    result = split_workflow(bpmn_xml)

    # ----- main workflow -----
    main_nodes, main_flows = _fingerprint_process(result.main_xml)

    exp_nodes = [tuple(n) for n in expected["main_nodes"]]
    exp_flows = [tuple(f) for f in expected["main_flows"]]

    assert main_nodes == exp_nodes, (
        f"\n[{case_name}] main NODE structure mismatch.\n"
        f"  expected ({len(exp_nodes)}): {exp_nodes}\n"
        f"  actual   ({len(main_nodes)}): {main_nodes}"
    )
    assert main_flows == exp_flows, (
        f"\n[{case_name}] main FLOW structure mismatch.\n"
        f"  expected ({len(exp_flows)}): {exp_flows}\n"
        f"  actual   ({len(main_flows)}): {main_flows}"
    )

    # ----- fragments -----
    exp_frags = expected["fragments"]
    assert len(result.fragments) == len(exp_frags), (
        f"[{case_name}] fragment count: expected {len(exp_frags)}, got {len(result.fragments)}"
    )

    for frag, exp in zip(result.fragments, exp_frags):
        assert frag.fragment_id == exp["fragment_id"], (
            f"[{case_name}] fragment_id: expected {exp['fragment_id']}, got {frag.fragment_id}"
        )
        assert frag.process_id == exp["process_id"], (
            f"[{case_name}] process_id mismatch on {frag.fragment_id}"
        )
        assert frag.wrapper_id == exp["wrapper_id"], (
            f"[{case_name}] wrapper_id mismatch on {frag.fragment_id}"
        )
        assert frag.input_variables == exp["inputs"], (
            f"[{case_name}] {frag.fragment_id} input vars: "
            f"expected {exp['inputs']}, got {frag.input_variables}"
        )
        assert frag.output_variables == exp["outputs"], (
            f"[{case_name}] {frag.fragment_id} output vars: "
            f"expected {exp['outputs']}, got {frag.output_variables}"
        )

        actual_task_ids = _fragment_task_ids(frag.xml)
        assert actual_task_ids == exp["task_ids"], (
            f"\n[{case_name}] {frag.fragment_id} task ids:\n"
            f"  expected: {exp['task_ids']}\n"
            f"  actual:   {actual_task_ids}"
        )

        actual_flow_ids = _fragment_flow_ids(frag.xml)
        assert actual_flow_ids == exp["flow_ids"], (
            f"\n[{case_name}] {frag.fragment_id} flow ids:\n"
            f"  expected: {exp['flow_ids']}\n"
            f"  actual:   {actual_flow_ids}"
        )

@pytest.mark.parametrize("case_name", CASES)
def test_outputs_are_well_formed_xml(case_name: str):
    """Every produced .bpmn file must parse as XML."""
    expected = EXPECTED[case_name]
    bpmn_xml = (BPMN_DIR / case_name).read_text(encoding="utf-8")
    if "nsup" in expected:
        pytest.skip("split not supported")
    result = split_workflow(bpmn_xml)
    ET.fromstring(result.main_xml)
    for frag in result.fragments:
        ET.fromstring(frag.xml)


def test_canonical_tc01_main_shape():
    bpmn_xml = (BPMN_DIR / "tc01_exec_before_adhoc.bpmn").read_text()
    result = split_workflow(bpmn_xml)

    assert len(result.fragments) == 1
    f = result.fragments[0]
    assert f.fragment_id == "E1"
    assert f.wrapper_id == "AdHoc_E1_Wrapper"

    nodes, flows = _fingerprint_process(result.main_xml)
    kinds = [n[0] for n in nodes]
    assert kinds == [
        "startEvent",
        "adHocSubProcess[wrapper=E1]",
        "adHocSubProcess[original]",
        "endEvent",
    ]
    assert len(flows) == 3
    assert flows[0][1] == "StartEvent_1"
    assert flows[0][2] == "AdHoc_E1_Wrapper"
    assert flows[1][1] == "AdHoc_E1_Wrapper"

    task_ids = _fragment_task_ids(f.xml)
    assert task_ids == ["UserTask_Input", "QHanaTask_Prepare", "QHanaTask_Analyze"]


def test_canonical_tc45_wu_palmer():
    bpmn_xml = (BPMN_DIR / "tc45_wu_palmer_partial.bpmn").read_text()
    result = split_workflow(bpmn_xml)

    assert len(result.fragments) == 1
    f = result.fragments[0]
    task_ids = _fragment_task_ids(f.xml)
    assert task_ids == [
        "Activity_1do8hxs",
        "Activity_0tfwzt0",
        "Activity_1nnwor0",
        "Activity_0s7n3hs",
        "Activity_0itt0yf",
    ]

    nodes, _ = _fingerprint_process(result.main_xml)
    kinds = [n[0] for n in nodes]
    assert "adHocSubProcess[original]" in kinds
    assert "adHocSubProcess[wrapper=E1]" in kinds


def test_canonical_tc20_nested_adhoc_raises():
    bpmn_xml = (BPMN_DIR / "tc20_nested_adhoc.bpmn").read_text()
    with pytest.raises(SplitNotSupported, match="[Nn]ested"):
        split_workflow(bpmn_xml)


def test_canonical_tc44_boundary_event_raises():
    bpmn_xml = (BPMN_DIR / "tc44_no_orphan_elements.bpmn").read_text()
    with pytest.raises(SplitNotSupported, match="boundaryEvent"):
        split_workflow(bpmn_xml)