import pprint
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "workflow_editor"))

try:
    from splitting.split import split_workflow, SplitNotSupported
except ImportError:
    if __name__ == "__main__":
        raise

BPMN = "http://www.omg.org/spec/BPMN/20100524/MODEL"
QHANA = "https://github.com/qhana"


def _ln(t):
    return t.split("}", 1)[1] if t.startswith("{") else t


def _find_proc(root):
    proc = root.find(f"{{{BPMN}}}process")
    if proc is None:
        for el in root.iter():
            if _ln(el.tag) == "process":
                return el
    return proc


def _fingerprint(xml_str):
    root = ET.fromstring(xml_str)
    proc = _find_proc(root)
    nodes, flows = [], []
    for child in proc:
        local = _ln(child.tag)
        if local == "sequenceFlow":
            ce = child.find(f"{{{BPMN}}}conditionExpression")
            cond = (ce.text or "").strip() if ce is not None else None
            flows.append(
                (child.get("id"), child.get("sourceRef"), child.get("targetRef"), cond)
            )
        elif local in ("laneSet", "association"):
            continue
        else:
            cid = child.get("id")
            if not cid:
                continue
            kind = local
            if local == "adHocSubProcess":
                fref = child.get(f"{{{QHANA}}}fragmentRef")
                kind = (
                    f"adHocSubProcess[wrapper={fref}]"
                    if fref
                    else "adHocSubProcess[original]"
                )
            nodes.append((kind, cid))
    return nodes, flows


def _task_ids(xml_str):
    root = ET.fromstring(xml_str)
    proc = _find_proc(root)
    if proc is None:
        return []
    return [
        c.get("id")
        for c in proc
        if _ln(c.tag)
        not in ("sequenceFlow", "startEvent", "endEvent", "laneSet", "association")
        and c.get("id")
    ]


def _flow_ids(xml_str):
    root = ET.fromstring(xml_str)
    proc = _find_proc(root)
    if proc is None:
        return []
    return [c.get("id") for c in proc if _ln(c.tag) == "sequenceFlow"]


def main():
    here = Path(__file__).parent
    bpmn_dir = here.parent / "workflow_editor" / "tests" / "bpmn"
    out_path = here / "expected_fixtures.py"

    fixtures = {}
    for p in sorted(bpmn_dir.glob("tc*.bpmn")):
        try:
            result = split_workflow(p.read_text())
        except SplitNotSupported as e:
            fixtures[p.name] = {"nsup": str(e)}
            continue
        mn, mf = _fingerprint(result.main_xml)
        frags = [
            {
                "fragment_id": f.fragment_id,
                "process_id": f.process_id,
                "wrapper_id": f.wrapper_id,
                "inputs": f.input_variables,
                "outputs": f.output_variables,
                "task_ids": _task_ids(f.xml),
                "flow_ids": _flow_ids(f.xml),
            }
            for f in result.fragments
        ]
        fixtures[p.name] = {
            "main_nodes": mn,
            "main_flows": mf,
            "fragments": frags,
        }

    with open(out_path, "w") as fh:
        fh.write('"""Auto-generated fixtures."""\n\nEXPECTED = ')
        fh.write(pprint.pformat(fixtures, width=120, sort_dicts=False))
        fh.write("\n")

    nsup = sum(1 for v in fixtures.values() if "nsup" in v)
    ok = len(fixtures) - nsup
    print(f"Regenerated {out_path}: {len(fixtures)} cases ({ok} OK, {nsup} NSUP)")


if __name__ == "__main__":
    main()
