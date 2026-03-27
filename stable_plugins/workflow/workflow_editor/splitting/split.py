from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import copy
import xml.etree.ElementTree as ET


BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"
OMGDC_NS = "http://www.omg.org/spec/DD/20100524/DC"
OMGDI_NS = "http://www.omg.org/spec/DD/20100524/DI"
CAMUNDA_NS = "http://camunda.org/schema/1.0/bpmn"
QHANA_NS = "https://github.com/qhana"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

NS = {
    "bpmn": BPMN_NS,
    "bpmndi": BPMNDI_NS,
    "omgdc": OMGDC_NS,
    "omgdi": OMGDI_NS,
    "camunda": CAMUNDA_NS,
    "qhana": QHANA_NS,
}

ET.register_namespace("", BPMN_NS)
ET.register_namespace("bpmndi", BPMNDI_NS)
ET.register_namespace("omgdc", OMGDC_NS)
ET.register_namespace("omgdi", OMGDI_NS)
ET.register_namespace("camunda", CAMUNDA_NS)
ET.register_namespace("qhana", QHANA_NS)
ET.register_namespace("xsi", XSI_NS)


@dataclass(frozen=True)
class BpmnNode:
    id: str
    tag: str
    elem: ET.Element


@dataclass(frozen=True)
class SequenceFlow:
    id: str
    source: str
    target: str
    elem: ET.Element


@dataclass(frozen=True)
class CompressedEdge:
    source: str
    target: str
    origin_flow_id: Optional[str] = None


class SplitNotSupported(Exception):
    """Raised when the workflow contains elements not yet supported by the splitter."""


def _localname(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _find_process(root: ET.Element, process_id: Optional[str] = None) -> ET.Element:
    processes = root.findall("bpmn:process", NS)
    if not processes:
        raise ValueError("No <bpmn:process> found in BPMN definitions!")

    if process_id is None:
        for p in processes:
            if (p.get("isExecutable") or "").lower() == "true":
                return p
        return processes[0]

    for p in processes:
        if p.get("id") == process_id:
            return p
    raise ValueError(f"Process with id={process_id!r} not found.")


def _extract_nodes_and_flows(
    process: ET.Element,
) -> Tuple[Dict[str, BpmnNode], Dict[str, SequenceFlow], str, List[str]]:
    nodes: Dict[str, BpmnNode] = {}
    flows: Dict[str, SequenceFlow] = {}

    start_id: Optional[str] = None
    end_ids: List[str] = []

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

        if ln in {
            "startEvent",
            "endEvent",
            "task",
            "serviceTask",
            "userTask",
            "manualTask",
            "scriptTask",
            "businessRuleTask",
            "sendTask",
            "receiveTask",
            "exclusiveGateway",
            "parallelGateway",
            "adHocSubProcess",
        } or child.tag == f"{{{QHANA_NS}}}qHAnaServiceTask":
            nodes[nid] = BpmnNode(id=nid, tag=child.tag, elem=child)

        if ln == "startEvent":
            if start_id is not None and start_id != nid:
                raise SplitNotSupported("Multiple startEvents are not supported yet.")
            start_id = nid
        elif ln == "endEvent":
            end_ids.append(nid)

    if not start_id or not end_ids:
        raise ValueError("Process must contain a startEvent and at least one endEvent.")

    return nodes, flows, start_id, end_ids


def _is_executable_qhana_task(node: BpmnNode) -> bool:
    if node.tag == f"{{{QHANA_NS}}}qHAnaServiceTask":
        return True

    ln = _localname(node.tag)
    if ln not in {"serviceTask", "task"}:
        return False

    cam_type = node.elem.get(f"{{{CAMUNDA_NS}}}type")
    cam_topic = node.elem.get(f"{{{CAMUNDA_NS}}}topic")
    return (cam_type == "external") and (cam_topic == "qhana-task")


def _validate_adhoc_subprocess_policy(process_elem: ET.Element) -> None:
    def _is_qhana_task_elem(e: ET.Element) -> bool:
        if e.tag == f"{{{QHANA_NS}}}qHAnaServiceTask":
            return True
        if e.tag in {f"{{{BPMN_NS}}}serviceTask", f"{{{BPMN_NS}}}task"}:
            cam_type = e.get(f"{{{CAMUNDA_NS}}}type")
            cam_topic = e.get(f"{{{CAMUNDA_NS}}}topic")
            return (cam_type == "external") and (cam_topic == "qhana-task")
        return False

    def _is_any_task_like(e: ET.Element) -> bool:
        if e.tag.startswith(f"{{{BPMN_NS}}}"):
            ln = _localname(e.tag)
            return ln in {
                "task",
                "serviceTask",
                "userTask",
                "manualTask",
                "scriptTask",
                "businessRuleTask",
                "sendTask",
                "receiveTask",
            }
        if e.tag == f"{{{QHANA_NS}}}qHAnaServiceTask":
            return True
        return False

    for adhoc in process_elem.findall("bpmn:adHocSubProcess", NS):
        adhoc_id = adhoc.get("id", "<no-id>")
        for child in adhoc.iter():
            if child is adhoc:
                continue
            if _is_any_task_like(child) and not _is_qhana_task_elem(child):
                raise SplitNotSupported(
                    f"adHocSubProcess '{adhoc_id}' contains non-QHAna task "
                    f"'{child.get('id', '<no-id>')}' ({child.tag})."
                )


def _build_adjacency(
    flows: Dict[str, SequenceFlow],
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, List[Tuple[str, str]]]]:
    outgoing: Dict[str, List[Tuple[str, str]]] = {}
    incoming: Dict[str, List[Tuple[str, str]]] = {}

    for f in flows.values():
        outgoing.setdefault(f.source, []).append((f.id, f.target))
        incoming.setdefault(f.target, []).append((f.id, f.source))

    return outgoing, incoming


def _validate_supported_nodes(nodes: Dict[str, BpmnNode]) -> None:
    unsupported = {
        "inclusiveGateway",
        "eventBasedGateway",
        "subProcess",
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
                f"Unsupported element {ln} id={n.id}. Currently supported: "
                "start/end, sequenceFlow, task/serviceTask/userTask/manualTask/scriptTask/"
                "businessRuleTask/sendTask/receiveTask, qHAnaServiceTask, "
                "exclusiveGateway, parallelGateway, adHocSubProcess."
            )


def _included_nodes_for_split(
    nodes: Dict[str, BpmnNode],
    start_id: str,
    end_ids: List[str],
    include_exec_tasks: bool,
) -> Set[str]:
    included: Set[str] = {start_id, *end_ids}

    rest_task_like = {
        "userTask",
        "manualTask",
        "scriptTask",
        "businessRuleTask",
        "sendTask",
        "receiveTask",
        "serviceTask",
        "task",
    }

    for nid, n in nodes.items():
        ln = _localname(n.tag)

        if ln in {"exclusiveGateway", "parallelGateway"}:
            included.add(nid)
            continue

        if ln == "adHocSubProcess":
            if not include_exec_tasks:
                included.add(nid)
            continue

        is_exec = _is_executable_qhana_task(n)

        if include_exec_tasks:
            if is_exec:
                included.add(nid)
        else:
            if (not is_exec) and ln not in {"startEvent", "endEvent"}:
                if ln in rest_task_like:
                    included.add(nid)

    return included


def _skip_to_next_included(
    start: str,
    outgoing: Dict[str, List[Tuple[str, str]]],
    included: Set[str],
    end_ids: Set[str],
    max_hops: int = 10_000,
) -> List[Tuple[str, str]]:
    next_included: List[Tuple[str, str]] = []

    for first_flow_id, first_target in outgoing.get(start, []):
        cur = first_target
        seen: Set[str] = set()
        hops = 0

        while cur not in included:
            if cur in end_ids:
                break

            if cur in seen:
                raise SplitNotSupported("Cycle detected while compressing split graph.")
            seen.add(cur)

            hops += 1
            if hops > max_hops:
                raise SplitNotSupported("Graph too large / possible cycle while compressing.")

            nxts = outgoing.get(cur, [])
            if len(nxts) == 0:
                cur = None  # type: ignore[assignment]
                break

            if len(nxts) > 1:
                raise SplitNotSupported(
                    f"Branching detected at excluded node {cur}. "
                    "Expected gateways to be included."
                )

            _, only_target = nxts[0]
            cur = only_target

        if cur is None:
            continue
        if cur in included:
            next_included.append((cur, first_flow_id))

    return next_included


def _build_compressed_edges(
    start_id: str,
    flows: Dict[str, SequenceFlow],
    included: Set[str],
    end_ids: List[str],
) -> List[CompressedEdge]:
    outgoing, _ = _build_adjacency(flows)
    end_set = set(end_ids)

    edges: List[CompressedEdge] = []

    for nid in list(included):
        for nxt, origin_flow_id in _skip_to_next_included(nid, outgoing, included, end_set):
            if nid != nxt:
                edges.append(CompressedEdge(source=nid, target=nxt, origin_flow_id=origin_flow_id))

    adj: Dict[str, List[str]] = {}
    for e in edges:
        adj.setdefault(e.source, []).append(e.target)

    reachable: Set[str] = set()
    stack = [start_id]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in adj.get(cur, []):
            if nxt not in reachable:
                stack.append(nxt)

    return [e for e in edges if e.source in reachable and e.target in reachable]


def _copy_node_without_io(node_elem: ET.Element) -> ET.Element:
    e = copy.deepcopy(node_elem)
    for child in list(e):
        if _localname(child.tag) in {"incoming", "outgoing"}:
            e.remove(child)
    return e


def _filter_lane(lane: ET.Element, surviving_node_ids: Set[str]) -> Optional[ET.Element]:
    lane_copy = copy.deepcopy(lane)

    for child in list(lane_copy):
        ln = _localname(child.tag)
        if ln == "flowNodeRef" or ln == "childLaneSet":
            lane_copy.remove(child)

    kept_refs = []
    for ref in lane.findall("bpmn:flowNodeRef", NS):
        ref_text = (ref.text or "").strip()
        if ref_text in surviving_node_ids:
            kept_refs.append(ref_text)

    for ref_id in kept_refs:
        ref_elem = ET.Element(f"{{{BPMN_NS}}}flowNodeRef")
        ref_elem.text = ref_id
        lane_copy.append(ref_elem)

    child_lane_set = lane.find("bpmn:childLaneSet", NS)
    if child_lane_set is not None:
        new_child_lane_set = ET.Element(child_lane_set.tag, attrib=dict(child_lane_set.attrib))
        for child_lane in child_lane_set.findall("bpmn:lane", NS):
            filtered = _filter_lane(child_lane, surviving_node_ids)
            if filtered is not None:
                new_child_lane_set.append(filtered)
        if len(list(new_child_lane_set)) > 0:
            lane_copy.append(new_child_lane_set)

    has_refs = len(lane_copy.findall("bpmn:flowNodeRef", NS)) > 0
    has_child_lanes = lane_copy.find("bpmn:childLaneSet", NS) is not None
    if has_refs or has_child_lanes:
        return lane_copy
    return None


def _copy_filtered_lane_sets(
    original_process: ET.Element,
    new_process: ET.Element,
    surviving_node_ids: Set[str],
) -> None:
    for lane_set in original_process.findall("bpmn:laneSet", NS):
        lane_set_copy = ET.Element(lane_set.tag, attrib=dict(lane_set.attrib))
        for lane in lane_set.findall("bpmn:lane", NS):
            filtered = _filter_lane(lane, surviving_node_ids)
            if filtered is not None:
                lane_set_copy.append(filtered)
        if len(list(lane_set_copy)) > 0:
            new_process.append(lane_set_copy)


def _collect_process_artifact_elements(original_process: ET.Element) -> Dict[str, ET.Element]:
    artifacts: Dict[str, ET.Element] = {}
    artifact_tags = {"dataObjectReference", "dataStoreReference", "textAnnotation", "group"}
    for child in list(original_process):
        ln = _localname(child.tag)
        cid = child.get("id")
        if cid and ln in artifact_tags:
            artifacts[cid] = child
    return artifacts


def _copy_relevant_artifacts_and_associations(
    original_process: ET.Element,
    new_process: ET.Element,
    surviving_node_ids: Set[str],
    surviving_flow_ids: Set[str],
) -> None:
    artifacts = _collect_process_artifact_elements(original_process)
    associations = [c for c in list(original_process) if _localname(c.tag) == "association"]

    directly_referenced_artifacts: Set[str] = set()
    for assoc in associations:
        src = assoc.get("sourceRef")
        tgt = assoc.get("targetRef")
        endpoints = {src, tgt}
        if any(e in surviving_node_ids or e in surviving_flow_ids for e in endpoints):
            for e in endpoints:
                if e in artifacts:
                    directly_referenced_artifacts.add(e)

    kept_artifact_ids: Set[str] = set()
    for aid in directly_referenced_artifacts:
        if aid in artifacts:
            new_process.append(copy.deepcopy(artifacts[aid]))
            kept_artifact_ids.add(aid)

    valid_refs = set(surviving_node_ids) | set(surviving_flow_ids) | kept_artifact_ids
    for assoc in associations:
        src = assoc.get("sourceRef")
        tgt = assoc.get("targetRef")
        if src in valid_refs and tgt in valid_refs:
            new_process.append(copy.deepcopy(assoc))


def _copy_collaboration_and_participants(
    root: ET.Element,
    original_root: ET.Element,
    original_process: ET.Element,
    new_process_id: str,
) -> None:
    """
    Copy collaborations from the original definitions tree, because root has
    already had its original collaboration elements removed.
    """
    original_process_id = original_process.get("id")
    if not original_process_id:
        return

    for collab in original_root.findall("bpmn:collaboration", NS):
        collab_copy = copy.deepcopy(collab)
        changed = False

        for participant in collab_copy.findall("bpmn:participant", NS):
            if participant.get("processRef") == original_process_id:
                participant.set("processRef", new_process_id)
                changed = True

        if changed:
            root.append(collab_copy)


def _copy_filtered_di(
    original_root: ET.Element,
    new_root: ET.Element,
    surviving_ids: Set[str],
    original_process_id: str,
    new_process_id: str,
) -> None:
    for diagram in original_root.findall("bpmndi:BPMNDiagram", NS):
        diagram_copy = copy.deepcopy(diagram)

        for plane in diagram_copy.findall("bpmndi:BPMNPlane", NS):
            if plane.get("bpmnElement") == original_process_id:
                plane.set("bpmnElement", new_process_id)

            for child in list(plane):
                ref = child.get("bpmnElement")
                if ref and ref not in surviving_ids and ref not in {original_process_id, new_process_id}:
                    plane.remove(child)

        new_root.append(diagram_copy)


def _make_process_from_subgraph(
    original_root: ET.Element,
    original_process: ET.Element,
    nodes: Dict[str, BpmnNode],
    included: Set[str],
    edges: List[CompressedEdge],
    new_process_id: str,
    new_process_name: str,
) -> ET.ElementTree:
    root = copy.deepcopy(original_root)

    for p in root.findall("bpmn:process", NS):
        root.remove(p)
    for c in root.findall("bpmn:collaboration", NS):
        root.remove(c)
    for d in root.findall("bpmndi:BPMNDiagram", NS):
        root.remove(d)

    orig_flows_by_pair: Dict[Tuple[str, str], List[ET.Element]] = {}
    orig_flows_by_id: Dict[str, ET.Element] = {}
    for sf in original_process.findall("bpmn:sequenceFlow", NS):
        sid = sf.get("id")
        src = sf.get("sourceRef")
        tgt = sf.get("targetRef")
        if sid:
            orig_flows_by_id[sid] = sf
        if src and tgt:
            orig_flows_by_pair.setdefault((src, tgt), []).append(sf)

    used_original_flow_ids: Set[str] = set()

    proc = copy.deepcopy(original_process)
    proc.set("id", new_process_id)
    proc.set("name", new_process_name)
    proc.set("isExecutable", "true")

    for c in list(proc):
        proc.remove(c)

    _copy_filtered_lane_sets(original_process, proc, included)

    included_nodes = [nid for nid in included if nid in nodes]
    starts = [nid for nid in included_nodes if _localname(nodes[nid].tag) == "startEvent"]
    ends = [nid for nid in included_nodes if _localname(nodes[nid].tag) == "endEvent"]
    middle = sorted([nid for nid in included_nodes if nid not in set(starts) and nid not in set(ends)])
    ordered = starts + middle + sorted(ends)

    for nid in ordered:
        proc.append(_copy_node_without_io(nodes[nid].elem))

    edge_list = sorted(edges, key=lambda e: (e.source, e.target, e.origin_flow_id or ""))

    created_flow_ids: Set[str] = set()
    flow_id_remap: Dict[str, str] = {}

    for i, edge in enumerate(edge_list, start=1):
        src, tgt = edge.source, edge.target

        chosen_original: Optional[ET.Element] = None
        for candidate in orig_flows_by_pair.get((src, tgt), []):
            cid = candidate.get("id")
            if cid and cid not in used_original_flow_ids:
                chosen_original = candidate
                break

        if chosen_original is not None:
            sf = copy.deepcopy(chosen_original)
            flow_id = sf.get("id")
            if flow_id:
                used_original_flow_ids.add(flow_id)
                flow_id_remap[flow_id] = flow_id
        else:
            flow_id = f"SplitFlow_{new_process_id}_{i}"
            sf = ET.Element(
                f"{{{BPMN_NS}}}sequenceFlow",
                attrib={"id": flow_id, "sourceRef": src, "targetRef": tgt},
            )

            if edge.origin_flow_id and edge.origin_flow_id in orig_flows_by_id:
                orig = orig_flows_by_id[edge.origin_flow_id]
                if orig.get("name"):
                    sf.set("name", orig.get("name"))
                cond = orig.find("bpmn:conditionExpression", NS)
                if cond is not None:
                    sf.append(copy.deepcopy(cond))
                flow_id_remap[edge.origin_flow_id] = flow_id

        sf.set("sourceRef", src)
        sf.set("targetRef", tgt)

        proc.append(sf)
        created_flow_ids.add(flow_id)

        src_elem = proc.find(f".//*[@id='{src}']")
        if src_elem is not None:
            out = ET.Element(f"{{{BPMN_NS}}}outgoing")
            out.text = flow_id
            src_elem.append(out)

        tgt_elem = proc.find(f".//*[@id='{tgt}']")
        if tgt_elem is not None:
            inc = ET.Element(f"{{{BPMN_NS}}}incoming")
            inc.text = flow_id
            tgt_elem.append(inc)

    for gw in proc.findall("bpmn:exclusiveGateway", NS) + proc.findall("bpmn:inclusiveGateway", NS):
        default_id = gw.get("default")
        if not default_id:
            continue

        if default_id in flow_id_remap:
            gw.set("default", flow_id_remap[default_id])
            default_id = gw.get("default")

        if default_id and default_id not in created_flow_ids:
            gw.attrib.pop("default", None)

    _copy_relevant_artifacts_and_associations(
        original_process=original_process,
        new_process=proc,
        surviving_node_ids=included,
        surviving_flow_ids=created_flow_ids,
    )

    root.append(proc)

    _copy_collaboration_and_participants(
        root=root,
        original_root=original_root,
        original_process=original_process,
        new_process_id=new_process_id,
    )

    surviving_ids = set(included) | set(created_flow_ids)
    for child in list(proc):
        cid = child.get("id")
        if cid:
            surviving_ids.add(cid)

    _copy_filtered_di(
        original_root=original_root,
        new_root=root,
        surviving_ids=surviving_ids,
        original_process_id=original_process.get("id", ""),
        new_process_id=new_process_id,
    )

    return ET.ElementTree(root)


def split_workflow(bpmn_xml: str, process_id: Optional[str] = None) -> Tuple[str, str]:
    original_root = ET.fromstring(bpmn_xml)
    original_process = _find_process(original_root, process_id=process_id)

    _validate_adhoc_subprocess_policy(original_process)

    nodes, flows, start_id, end_ids = _extract_nodes_and_flows(original_process)
    _validate_supported_nodes(nodes)

    exec_included = _included_nodes_for_split(nodes, start_id, end_ids, include_exec_tasks=True)
    exec_edges = _build_compressed_edges(start_id, flows, exec_included, end_ids)

    rest_included = _included_nodes_for_split(nodes, start_id, end_ids, include_exec_tasks=False)
    rest_edges = _build_compressed_edges(start_id, flows, rest_included, end_ids)

    exec_tree = _make_process_from_subgraph(
        original_root=original_root,
        original_process=original_process,
        nodes=nodes,
        included=exec_included,
        edges=exec_edges,
        new_process_id=f"{original_process.get('id')}_exec",
        new_process_name=f"{original_process.get('name', 'process')} (exec split)",
    )

    rest_tree = _make_process_from_subgraph(
        original_root=original_root,
        original_process=original_process,
        nodes=nodes,
        included=rest_included,
        edges=rest_edges,
        new_process_id=f"{original_process.get('id')}_rest",
        new_process_name=f"{original_process.get('name', 'process')} (rest split)",
    )

    exec_xml_out = ET.tostring(exec_tree.getroot(), encoding="utf-8", xml_declaration=True).decode("utf-8")
    rest_xml_out = ET.tostring(rest_tree.getroot(), encoding="utf-8", xml_declaration=True).decode("utf-8")
    return exec_xml_out, rest_xml_out