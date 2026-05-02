import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import pytest
    from splitting.split import split_workflow, SplitNotSupported, write_split_outputs
    from expected_fixtures import EXPECTED
except ImportError:
    if __name__ == "__main__":
        raise

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
QHANA_NS = "https://github.com/qhana"
NS = {"bpmn": BPMN_NS, "qhana": QHANA_NS}

ROOT = Path(__file__).parent
BPMN_DIR = ROOT.parent / "workflow_editor" / "tests" / "bpmn"
OUT_DIR = ROOT / "_out"


def _ln(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _fingerprint_process(
    xml_str: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[Any, ...]]]:
    root = ET.fromstring(xml_str)
    proc = root.find(f"{{{BPMN_NS}}}process")
    if proc is None:
        proc = root.find(".//bpmn:process", NS)
    assert proc is not None

    nodes: List[Tuple[str, str]] = []
    flows: List[Tuple[Any, ...]] = []
    for child in proc:
        local = _ln(child.tag)
        if local == "sequenceFlow":
            cond_elem = child.find("bpmn:conditionExpression", NS)
            cond = (cond_elem.text or "").strip() if cond_elem is not None else None
            flows.append(
                (
                    child.get("id"),
                    child.get("sourceRef"),
                    child.get("targetRef"),
                    cond,
                )
            )
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
    root = ET.fromstring(xml_str)
    proc = root.find(f"{{{BPMN_NS}}}process")
    assert proc is not None
    out: List[str] = []
    for child in proc:
        local = _ln(child.tag)
        if local in (
            "sequenceFlow",
            "startEvent",
            "endEvent",
            "laneSet",
            "association",
        ):
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
        with pytest.raises(SplitNotSupported):
            split_workflow(bpmn_xml)
        return

    stem = Path(case_name).stem
    paths = write_split_outputs(bpmn_xml, OUT_DIR, stem)
    assert len(paths) >= 1

    result = split_workflow(bpmn_xml)

    main_nodes, main_flows = _fingerprint_process(result.main_xml)
    exp_nodes = [tuple(n) for n in expected["main_nodes"]]
    exp_flows = [tuple(f) for f in expected["main_flows"]]

    assert main_nodes == exp_nodes, (
        f"\n[{case_name}] main NODE mismatch.\n"
        f"  expected ({len(exp_nodes)}): {exp_nodes}\n"
        f"  actual   ({len(main_nodes)}): {main_nodes}"
    )
    assert main_flows == exp_flows, (
        f"\n[{case_name}] main FLOW mismatch.\n"
        f"  expected ({len(exp_flows)}): {exp_flows}\n"
        f"  actual   ({len(main_flows)}): {main_flows}"
    )

    exp_frags = expected["fragments"]
    assert len(result.fragments) == len(exp_frags), (
        f"[{case_name}] fragment count: "
        f"expected {len(exp_frags)}, got {len(result.fragments)}"
    )

    for frag, exp in zip(result.fragments, exp_frags):
        assert frag.fragment_id == exp["fragment_id"]
        assert frag.process_id == exp["process_id"]
        assert frag.wrapper_id == exp["wrapper_id"]
        assert frag.input_variables == exp["inputs"]
        assert frag.output_variables == exp["outputs"]

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
    expected = EXPECTED[case_name]
    bpmn_xml = (BPMN_DIR / case_name).read_text(encoding="utf-8")
    if "nsup" in expected:
        with pytest.raises(SplitNotSupported):
            split_workflow(bpmn_xml)
        return
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

    nodes, _ = _fingerprint_process(result.main_xml)
    kinds = [n[0] for n in nodes]
    assert "startEvent" in kinds
    assert "adHocSubProcess[wrapper=E1]" in kinds
    assert "adHocSubProcess[original]" in kinds


def test_canonical_tc20_nested_adhoc_rejected():
    bpmn_xml = (BPMN_DIR / "tc20_nested_adhoc.bpmn").read_text()
    with pytest.raises(SplitNotSupported, match="[Nn]ested"):
        split_workflow(bpmn_xml)


def test_canonical_tc21_mixed_adhoc_rejected():
    bpmn_xml = (BPMN_DIR / "tc21_adhoc_with_non_qhana_task.bpmn").read_text()
    with pytest.raises(SplitNotSupported, match="non-QHAna"):
        split_workflow(bpmn_xml)


def test_canonical_tc43_mixed_adhoc_rejected():
    bpmn_xml = (BPMN_DIR / "tc43_start_to_end_path.bpmn").read_text()
    with pytest.raises(SplitNotSupported, match="non-QHAna"):
        split_workflow(bpmn_xml)


def test_canonical_tc44_fully_extracted():
    bpmn_xml = (BPMN_DIR / "tc44_no_orphan_elements.bpmn").read_text()
    result = split_workflow(bpmn_xml)
    assert len(result.fragments) == 1
    f = result.fragments[0]
    root = ET.fromstring(f.xml)
    ids = {e.get("id") for e in root.iter() if e.get("id")}
    assert "Task_Exec" in ids
    assert "BoundaryEvent_1" in ids


def test_canonical_tc45_wu_palmer():
    bpmn_xml = (BPMN_DIR / "tc45_wu_palmer_partial.bpmn").read_text()
    result = split_workflow(bpmn_xml)

    assert len(result.fragments) == 1

    nodes, _ = _fingerprint_process(result.main_xml)
    kinds = [n[0] for n in nodes]
    assert "adHocSubProcess[original]" in kinds
    assert "adHocSubProcess[wrapper=E1]" in kinds
