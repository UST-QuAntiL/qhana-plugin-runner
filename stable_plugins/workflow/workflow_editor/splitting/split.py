from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import copy
import xml.etree.ElementTree as ET


BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
CAMUNDA_NS = "http://camunda.org/schema/1.0/bpmn"
QHANA_NS = "https://github.com/qhana"

NS = {
    "bpmn": BPMN_NS,
    "camunda": CAMUNDA_NS,
}


@dataclass(frozen=True)
class BpmnNode:
    id: str
    tag: str  # full tag with namespace
    elem: ET.Element


@dataclass(frozen=True)
class SequenceFlow:
    id: str
    source: str
    target: str
    elem: ET.Element


class SplitNotSupported(Exception):
    """Raised when the workflow contains elements not yet supported by the splitter."""


def _localname(tag: str) -> str:
    """'{ns}task' -> 'task' """
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _find_process(root: ET.Element, process_id: Optional[str] = None) -> ET.Element:
    processes = root.findall("bpmn:process", NS)
    if not processes:
        raise ValueError("No <bpmn:process> found in BPMN definitions!")

    if process_id is None:
        # pick first executable process if possible, else first process
        for p in processes:
            if (p.get("isExecutable") or "").lower() == "true":
                return p
        return processes[0]

    for p in processes:
        if p.get("id") == process_id:
            return p
    raise ValueError(f"Process with id={process_id!r} not found.")


def _extract_nodes_and_flows(process: ET.Element) -> Tuple[Dict[str, BpmnNode], Dict[str, SequenceFlow], str, str]:
    """
    Returns:
      nodes_by_id, flows_by_id, start_event_id, end_event_id
    """
    nodes: Dict[str, BpmnNode] = {}
    flows: Dict[str, SequenceFlow] = {}

    start_id: Optional[str] = None
    end_id: Optional[str] = None

    for child in list(process):
        ln = _localname(child.tag)

        if ln == "sequenceFlow":
            fid = child.get("id")
            src = child.get("sourceRef")
            tgt = child.get("targetRef")
            if not fid or not src or not tgt:
                continue
            flows[fid] = SequenceFlow(id=fid, source=src, target=tgt, elem=child)
            continue

        nid = child.get("id")
        if not nid:
            continue

        nodes[nid] = BpmnNode(id=nid, tag=child.tag, elem=child)

        if ln == "startEvent":
            start_id = nid
        elif ln == "endEvent":
            end_id = nid

    if not start_id or not end_id:
        raise ValueError("Process must contain a startEvent and endEvent for this splitter v0.")

    return nodes, flows, start_id, end_id


def _is_supported_linear_subset(nodes: Dict[str, BpmnNode]) -> None:
    """
    v0 limitation: no gateways, no subprocesses, no boundary events, etc.
    We'll extend later.
    """
    unsupported = {
        "exclusiveGateway",
        "inclusiveGateway",
        "parallelGateway",
        "eventBasedGateway",
        "subProcess",
        # "adHocSubProcess",
        "callActivity",
        "transaction",
        "boundaryEvent",
        "intermediateCatchEvent",
        "intermediateThrowEvent",
    }
    for n in nodes.values():
        ln = _localname(n.tag)
        if ln in unsupported:
            raise SplitNotSupported(
                f"Splitter v0 supports only linear workflows (no {ln}). "
                f"Found unsupported element id={n.id}."
            )


def _is_executable_qhana_task(node: BpmnNode) -> bool:
    """
    Executable nodes:
      A) Wu-Palmer style transformed tasks:
         <bpmn:serviceTask camunda:type="external" camunda:topic="qhana-task" .../>
      B) QHAna workflow-editor native tasks:
         <qhana:qHAnaServiceTask .../>
    """
    # B) native qhana element
    if node.tag == f"{{{QHANA_NS}}}qHAnaServiceTask":
        return True

    # A) transformed camunda external task
    ln = _localname(node.tag)
    if ln not in {"serviceTask", "task"}:
        return False

    cam_type = node.elem.get(f"{{{CAMUNDA_NS}}}type")
    cam_topic = node.elem.get(f"{{{CAMUNDA_NS}}}topic")
    return (cam_type == "external") and (cam_topic == "qhana-task")


def _flatten_adhoc_subprocess(process_elem):
    """
    Minimal support for bpmn:adHocSubProcess used as a grouping container.

    Strategy:
    - If an adHocSubProcess contains exactly one child "task-like" element,
      lift that element into the process.
    - Rewire sequenceFlows that pointed to/from the adHocSubProcess to point to/from the lifted child.
    - Remove the adHocSubProcess element.
    """

    BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
    QHANA_NS = "https://github.com/qhana"

    def _is_task_like(elem):
        # support both:
        # - transformed BPMN tasks: {BPMN}serviceTask/scriptTask/task
        # - raw qhana:qHAnaServiceTask
        if elem.tag.startswith(f"{{{BPMN_NS}}}"):
            ln = elem.tag.split("}", 1)[1]
            return ln in {"serviceTask", "scriptTask", "task"}
        if elem.tag == f"{{{QHANA_NS}}}qHAnaServiceTask":
            return True
        return False

    # Find all adHocSubProcess elements
    adhocs = [e for e in list(process_elem) if e.tag == f"{{{BPMN_NS}}}adHocSubProcess"]

    for adhoc in adhocs:
        adhoc_id = adhoc.attrib.get("id")
        if not adhoc_id:
            continue

        children = [c for c in adhoc.iter() if c is not adhoc and _is_task_like(c)]

        if len(children) != 1:
            continue

        child = children[0]
        child_id = child.attrib.get("id")
        if not child_id:
            continue

        idx = list(process_elem).index(adhoc)
        process_elem.insert(idx, copy.deepcopy(child))
        child_id = child.attrib.get("id")

        for sf in process_elem.findall(f".//{{{BPMN_NS}}}sequenceFlow"):
            if sf.attrib.get("sourceRef") == adhoc_id:
                sf.attrib["sourceRef"] = child_id
            if sf.attrib.get("targetRef") == adhoc_id:
                sf.attrib["targetRef"] = child_id

        process_elem.remove(adhoc)


def _follow_linear_path(
    start_id: str,
    end_id: str,
    flows: Dict[str, SequenceFlow],
) -> List[str]:
    """
    Follows the unique sequenceFlow path start -> ... -> end.
    Raises if branching/merging is detected.
    """
    outgoing: Dict[str, List[str]] = {}
    incoming: Dict[str, List[str]] = {}

    for f in flows.values():
        outgoing.setdefault(f.source, []).append(f.target)
        incoming.setdefault(f.target, []).append(f.source)

    path = [start_id]
    cur = start_id
    seen: Set[str] = {start_id}

    while cur != end_id:
        nxts = outgoing.get(cur, [])
        if len(nxts) == 0:
            raise SplitNotSupported(f"No outgoing flow from {cur}. Cannot reach endEvent.")
        if len(nxts) > 1:
            raise SplitNotSupported(f"Branching detected at {cur}. Splitter v0 is linear-only.")
        nxt = nxts[0]

        inc = incoming.get(nxt, [])
        if len(inc) > 1:
            raise SplitNotSupported(f"Merging detected at {nxt}. Splitter v0 is linear-only.")

        if nxt in seen:
            raise SplitNotSupported("Cycle detected. Splitter v0 is linear-only.")
        seen.add(nxt)
        path.append(nxt)
        cur = nxt

    return path


def _make_minimal_process_from_ids(
    original_root: ET.Element,
    original_process: ET.Element,
    node_ids_in_order: List[str],
    nodes: Dict[str, BpmnNode],
    new_process_id: str,
    new_process_name: str,
) -> ET.ElementTree:
    """
    Create a new BPMN definitions with a single process containing:
      startEvent, selected nodes, endEvent
    and sequence flows connecting them linearly.
    """
    root = copy.deepcopy(original_root)

    # Remove all existing processes from definitions
    for p in root.findall("bpmn:process", NS):
        root.remove(p)

    proc = copy.deepcopy(original_process)
    proc.set("id", new_process_id)
    proc.set("name", new_process_name)
    proc.set("isExecutable", "true")  

    # Clear children
    for c in list(proc):
        proc.remove(c)

    start_id = node_ids_in_order[0]
    end_id = node_ids_in_order[-1]

    # Copy start/end elements
    start_elem = copy.deepcopy(nodes[start_id].elem)
    end_elem = copy.deepcopy(nodes[end_id].elem)

    # have to set incoming/outgoing later, so clear them first to avoid stale refs
    for e in (start_elem, end_elem):
        for child in list(e):
            ln = _localname(child.tag)
            if ln in {"incoming", "outgoing"}:
                e.remove(child)

    proc.append(start_elem)

    selected_middle = node_ids_in_order[1:-1]
    for mid_id in selected_middle:
        mid_elem = copy.deepcopy(nodes[mid_id].elem)
        # Clear incoming/outgoing like above
        for child in list(mid_elem):
            ln = _localname(child.tag)
            if ln in {"incoming", "outgoing"}:
                mid_elem.remove(child)
        proc.append(mid_elem)

    proc.append(end_elem)

    # Create new flows connecting kept nodes
    chain = [start_id] + selected_middle + [end_id]

    # Add outgoing/incoming tags + sequenceFlows
    for i in range(len(chain) - 1):
        src = chain[i]
        tgt = chain[i + 1]
        flow_id = f"SplitFlow_{new_process_id}_{i+1}"

        # sequenceFlow
        sf = ET.Element(f"{{{BPMN_NS}}}sequenceFlow", attrib={"id": flow_id, "sourceRef": src, "targetRef": tgt})
        proc.append(sf)

        # outgoing on src
        src_elem = proc.find(f".//*[@id='{src}']")
        if src_elem is not None:
            out = ET.Element(f"{{{BPMN_NS}}}outgoing")
            out.text = flow_id
            src_elem.append(out)

        # incoming on tgt
        tgt_elem = proc.find(f".//*[@id='{tgt}']")
        if tgt_elem is not None:
            inc = ET.Element(f"{{{BPMN_NS}}}incoming")
            inc.text = flow_id
            tgt_elem.append(inc)

    root.append(proc)
    return ET.ElementTree(root)


def split_workflow(bpmn_xml: str, process_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Split BPMN into (executable_part_xml, remainder_part_xml).

    v0 assumptions:
      - Exactly one linear path from startEvent to endEvent (no gateways/merges/cycles)
      - Executable nodes are Camunda external tasks with topic 'qhana-task'
      - Everything else becomes remainder (except start/end which are duplicated)
    """
    original_root = ET.fromstring(bpmn_xml)
    original_process = _find_process(original_root, process_id=process_id)

    _flatten_adhoc_subprocess(original_process)

    nodes, flows, start_id, end_id = _extract_nodes_and_flows(original_process)

    _is_supported_linear_subset(nodes)

    path = _follow_linear_path(start_id, end_id, flows)

    # classify nodes along the path (excluding start/end)
    exec_middle: List[str] = []
    rest_middle: List[str] = []

    for nid in path[1:-1]:
        node = nodes[nid]
        if _is_executable_qhana_task(node):
            exec_middle.append(nid)
        else:
            rest_middle.append(nid)

    exec_path = [start_id] + exec_middle + [end_id]
    rest_path = [start_id] + rest_middle + [end_id]

    exec_tree = _make_minimal_process_from_ids(
        original_root=original_root,
        original_process=original_process,
        node_ids_in_order=exec_path,
        nodes=nodes,
        new_process_id=f"{original_process.get('id')}_exec",
        new_process_name=f"{original_process.get('name','process')} (exec split)",
    )
    rest_tree = _make_minimal_process_from_ids(
        original_root=original_root,
        original_process=original_process,
        node_ids_in_order=rest_path,
        nodes=nodes,
        new_process_id=f"{original_process.get('id')}_rest",
        new_process_name=f"{original_process.get('name','process')} (rest split)",
    )

    exec_xml_out = ET.tostring(exec_tree.getroot(), encoding="utf-8", xml_declaration=True).decode("utf-8")
    rest_xml_out = ET.tostring(rest_tree.getroot(), encoding="utf-8", xml_declaration=True).decode("utf-8")
    return exec_xml_out, rest_xml_out