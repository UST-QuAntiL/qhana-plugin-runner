from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Set, Tuple

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
    "xsi": XSI_NS,
}


class SplitNotSupported(Exception):
    """Raised when the input uses elements outside Phase 1 scope."""


@dataclass
class FragmentResult:
    fragment_id: str
    process_id: str
    wrapper_id: str
    xml: str
    input_variables: List[str] = field(default_factory=list)
    output_variables: List[str] = field(default_factory=list)


@dataclass
class SplitResult:
    main_xml: str
    fragments: List[FragmentResult] = field(default_factory=list)


class Classifier(Protocol):
    """Decides whether a top-level flow node is extractable at runtime."""

    def is_extractable(self, elem: ET.Element) -> bool: ...


def default_classifier(elem: ET.Element) -> bool:
    tag = elem.tag
    if tag == f"{{{QHANA_NS}}}qHAnaServiceTask":
        return True
    local = _localname(tag)
    if local == "serviceTask":
        ctype = elem.get(f"{{{CAMUNDA_NS}}}type")
        ctopic = elem.get(f"{{{CAMUNDA_NS}}}topic") or ""
        if ctype != "external":
            return False
        return (
            ctopic == "qhana-task"
            or ctopic.startswith("plugin.")
            or ctopic.startswith("plugin-step.")
        )
    if local in {"userTask", "task", "manualTask"}:
        return True
    return False


def _localname(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _bpmn(name: str) -> str:
    return f"{{{BPMN_NS}}}{name}"


def _qhana(name: str) -> str:
    return f"{{{QHANA_NS}}}{name}"


def _camunda(name: str) -> str:
    return f"{{{CAMUNDA_NS}}}{name}"


TASK_LOCALNAMES = {
    "task",
    "userTask",
    "serviceTask",
    "manualTask",
    "scriptTask",
    "businessRuleTask",
    "sendTask",
    "receiveTask",
}

UNSUPPORTED_LOCALNAMES = {
    "inclusiveGateway",
    "eventBasedGateway",
    "complexGateway",
    "subProcess",
    "callActivity",
    "transaction",
    "boundaryEvent",
    "intermediateCatchEvent",
    "intermediateThrowEvent",
}


def _is_task_like(elem: ET.Element) -> bool:
    if elem.tag == _qhana("qHAnaServiceTask"):
        return True
    local = _localname(elem.tag)
    return local in TASK_LOCALNAMES


def _strip_incoming_outgoing(elem: ET.Element) -> None:
    """Remove <bpmn:incoming> and <bpmn:outgoing> children in place."""
    for child in list(elem):
        if _localname(child.tag) in {"incoming", "outgoing"}:
            elem.remove(child)


def _add_incoming(elem: ET.Element, flow_id: str) -> None:
    e = ET.SubElement(elem, _bpmn("incoming"))
    e.text = flow_id


def _add_outgoing(elem: ET.Element, flow_id: str) -> None:
    e = ET.SubElement(elem, _bpmn("outgoing"))
    e.text = flow_id


@dataclass
class Node:
    id: str
    tag: str
    local: str
    elem: ET.Element
    extractable: bool = False


@dataclass
class Flow:
    id: str
    source: str
    target: str
    elem: ET.Element


@dataclass
class Region:
    index: int
    node_ids: List[str]
    entry_flow_ids: List[str]
    exit_flow_ids: List[str]
    internal_flow_ids: List[str]


def _find_process(root: ET.Element, process_id: Optional[str]) -> ET.Element:
    processes = root.findall(f"{{{BPMN_NS}}}process")
    if not processes:
        raise ValueError("No <bpmn:process> element found.")
    if process_id is not None:
        for p in processes:
            if p.get("id") == process_id:
                return p
        raise ValueError(f"Process id={process_id!r} not found.")
    for p in processes:
        if (p.get("isExecutable") or "").lower() == "true":
            return p
    return processes[0]


def _parse_process(
    process: ET.Element,
) -> Tuple[Dict[str, Node], Dict[str, Flow], List[str], str, List[str]]:
    nodes: Dict[str, Node] = {}
    flows: Dict[str, Flow] = {}
    order: List[str] = []
    start_id: Optional[str] = None
    end_ids: List[str] = []

    for child in list(process):
        local = _localname(child.tag)

        if local in {"laneSet", "documentation", "extensionElements", "ioSpecification"}:
            continue

        if local == "sequenceFlow":
            fid = child.get("id")
            src = child.get("sourceRef")
            tgt = child.get("targetRef")
            if fid and src and tgt:
                flows[fid] = Flow(id=fid, source=src, target=tgt, elem=child)
            continue

        if local in {
            "dataObject",
            "dataObjectReference",
            "dataStoreReference",
            "textAnnotation",
            "group",
            "association",
        }:
            continue

        if local in UNSUPPORTED_LOCALNAMES:
            raise SplitNotSupported(
                f"Unsupported Phase 2/3 element {local!r} (id={child.get('id')!r})."
            )

        nid = child.get("id")
        if not nid:
            continue

        if local == "adHocSubProcess":
            for descendant in child.iter():
                if descendant is child:
                    continue
                if _localname(descendant.tag) == "adHocSubProcess":
                    raise SplitNotSupported(
                        f"Nested ad-hoc subprocess not supported "
                        f"(outer id={nid!r}, inner id={descendant.get('id')!r})."
                    )

        nodes[nid] = Node(id=nid, tag=child.tag, local=local, elem=child)
        order.append(nid)

        if local == "startEvent":
            if start_id is not None:
                raise SplitNotSupported(
                    "Multiple start events are not supported in Phase 1."
                )
            start_id = nid
        elif local == "endEvent":
            end_ids.append(nid)

    if start_id is None:
        raise ValueError("Process has no start event.")
    if not end_ids:
        raise ValueError("Process has no end event.")

    return nodes, flows, order, start_id, end_ids


def _classify_nodes(
    nodes: Dict[str, Node],
    classifier: Callable[[ET.Element], bool],
) -> None:
    for node in nodes.values():
        if node.local in {
            "startEvent",
            "endEvent",
            "exclusiveGateway",
            "parallelGateway",
            "adHocSubProcess",
        }:
            node.extractable = False
            continue
        node.extractable = classifier(node.elem)


def _build_adjacency(
    flows: Dict[str, Flow],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (outgoing_flow_ids, incoming_flow_ids) indexed by node id."""
    out: Dict[str, List[str]] = {}
    inc: Dict[str, List[str]] = {}
    for f in flows.values():
        out.setdefault(f.source, []).append(f.id)
        inc.setdefault(f.target, []).append(f.id)
    return out, inc


def _find_regions(
    nodes: Dict[str, Node],
    flows: Dict[str, Flow],
    order: List[str],
) -> List[Region]:

    out_flows, in_flows = _build_adjacency(flows)

    def succ_of(nid: str) -> List[Tuple[str, str]]:
        """(target_id, flow_id) pairs for successors of nid."""
        return [(flows[fid].target, fid) for fid in out_flows.get(nid, [])]

    def pred_of(nid: str) -> List[Tuple[str, str]]:
        return [(flows[fid].source, fid) for fid in in_flows.get(nid, [])]

    assigned: Set[str] = set()
    regions: List[Region] = []
    idx = 0

    for nid in order:
        if nid in assigned:
            continue
        node = nodes[nid]
        if not node.extractable:
            continue

        cur = nid
        while True:
            preds = pred_of(cur)
            ext_preds = [
                (p, f) for (p, f) in preds if nodes.get(p) and nodes[p].extractable
            ]
            if len(ext_preds) != 1:
                break
            p, _f = ext_preds[0]
            p_succs = succ_of(p)
            p_ext_succs = [
                (t, fi) for (t, fi) in p_succs if nodes.get(t) and nodes[t].extractable
            ]
            if len(p_ext_succs) != 1 or p_ext_succs[0][0] != cur:
                break
            cur = p
        region_start = cur

        chain: List[str] = [region_start]
        internal_flow_ids: List[str] = []
        while True:
            last = chain[-1]
            succs = succ_of(last)
            ext_succs = [
                (t, f) for (t, f) in succs if nodes.get(t) and nodes[t].extractable
            ]
            if len(ext_succs) != 1:
                break
            t, fid = ext_succs[0]
            t_preds = pred_of(t)
            t_ext_preds = [
                (p, fi) for (p, fi) in t_preds if nodes.get(p) and nodes[p].extractable
            ]
            if len(t_ext_preds) != 1 or t_ext_preds[0][0] != last:
                break
            flows_last_to_t = [f for f in out_flows.get(last, []) if flows[f].target == t]
            if len(flows_last_to_t) != 1:
                break
            chain.append(t)
            internal_flow_ids.append(fid)

        chain_set = set(chain)
        entry_flows = [
            f for f in in_flows.get(chain[0], []) if flows[f].source not in chain_set
        ]
        exit_flows = [
            f for f in out_flows.get(chain[-1], []) if flows[f].target not in chain_set
        ]
        if not entry_flows:
            raise SplitNotSupported(
                f"Region starting at {chain[0]!r} has no entry flow from outside."
            )
        if not exit_flows:
            raise SplitNotSupported(
                f"Region ending at {chain[-1]!r} has no exit flow to outside."
            )
        entry_sources = {flows[f].source for f in entry_flows}
        exit_targets = {flows[f].target for f in exit_flows}
        if len(entry_sources) != 1:
            raise SplitNotSupported(
                f"Region starting at {chain[0]!r} has entry flows from "
                f"{len(entry_sources)} distinct sources; Phase 1 requires one."
            )
        if len(exit_targets) != 1:
            raise SplitNotSupported(
                f"Region ending at {chain[-1]!r} has exit flows to "
                f"{len(exit_targets)} distinct targets; Phase 1 requires one."
            )

        idx += 1
        regions.append(
            Region(
                index=idx,
                node_ids=chain,
                entry_flow_ids=entry_flows,
                exit_flow_ids=exit_flows,
                internal_flow_ids=internal_flow_ids,
            )
        )
        assigned.update(chain)

    return regions


def assigned_set(chain: List[str]) -> Set[str]:
    return set(chain)


VAR_PATTERN = re.compile(r"\$\{([a-zA-Z_][\w.]*)")


def _scan_vars_in_element(elem: ET.Element) -> Tuple[Set[str], Set[str]]:
    reads: Set[str] = set()
    writes: Set[str] = set()
    for io in elem.findall(f".//{_camunda('inputOutput')}"):
        for ip in io.findall(_camunda("inputParameter")):
            if ip.text:
                reads.update(VAR_PATTERN.findall(ip.text))
        for op in io.findall(_camunda("outputParameter")):
            if op.text:
                reads.update(VAR_PATTERN.findall(op.text))
            name = op.get("name")
            if name:
                writes.add(name)
    for ff in elem.findall(f".//{_camunda('formField')}"):
        fid = ff.get("id")
        if fid:
            writes.add(fid)
    return reads, writes


def _compute_region_io(
    region: Region,
    nodes: Dict[str, Node],
    all_regions: Optional[List["Region"]] = None,
) -> Tuple[List[str], List[str]]:
    reads: Set[str] = set()
    writes: Set[str] = set()
    for nid in region.node_ids:
        r, w = _scan_vars_in_element(nodes[nid].elem)
        reads |= r
        writes |= w
    reads.discard("output")

    if all_regions is not None:
        region_node_ids = set(region.node_ids)
        external_reads: Set[str] = set()
        for nid, node in nodes.items():
            if nid in region_node_ids:
                continue
            r2, _w2 = _scan_vars_in_element(node.elem)
            external_reads |= r2
        external_reads.discard("output")
        outputs_set = writes & external_reads
    else:
        outputs_set = set(writes)

    inputs = sorted(reads - writes)
    outputs = sorted(outputs_set)
    return inputs, outputs


def _clone_task_for_inner(elem: ET.Element) -> ET.Element:
    c = copy.deepcopy(elem)
    _strip_incoming_outgoing(c)
    return c


def _make_wrapper_adhoc(
    region: Region,
    nodes: Dict[str, Node],
    flows: Dict[str, Flow],
    all_regions: List["Region"],
    original_process_id: str,
) -> Tuple[ET.Element, List[str], List[str]]:
    fid = f"E{region.index}"
    wrapper_id = f"AdHoc_{fid}_Wrapper"
    inner_start_id = f"StartEvent_{wrapper_id}"
    inner_end_id = f"EndEvent_{wrapper_id}"
    plugin_task_id = f"ServiceTask_{fid}_Plugin"
    plugin_topic = f"plugin-step.{original_process_id}-{fid}"

    wrapper = ET.Element(
        _bpmn("adHocSubProcess"),
        attrib={
            "id": wrapper_id,
            "name": f"Extracted Fragment {fid}",
        },
    )
    wrapper.set(_qhana("fragmentRef"), fid)

    inputs, outputs = _compute_region_io(region, nodes, all_regions)

    inner_start = ET.SubElement(
        wrapper, _bpmn("startEvent"), attrib={"id": inner_start_id}
    )
    ET.SubElement(inner_start, _bpmn("outgoing")).text = f"Flow_{wrapper_id}_in"

    plugin_task = ET.SubElement(
        wrapper,
        _bpmn("serviceTask"),
        attrib={
            "id": plugin_task_id,
            "name": f"Extracted Fragment {fid}",
        },
    )
    plugin_task.set(_camunda("type"), "external")
    plugin_task.set(_camunda("topic"), plugin_topic)
    ET.SubElement(plugin_task, _bpmn("incoming")).text = f"Flow_{wrapper_id}_in"
    ET.SubElement(plugin_task, _bpmn("outgoing")).text = f"Flow_{wrapper_id}_out"

    if inputs or outputs:
        ext = ET.SubElement(plugin_task, _bpmn("extensionElements"))
        io = ET.SubElement(ext, _camunda("inputOutput"))
        for v in inputs:
            ip = ET.SubElement(io, _camunda("inputParameter"), attrib={"name": v})
            ip.text = "${" + v + "}"
        for v in outputs:
            op = ET.SubElement(io, _camunda("outputParameter"), attrib={"name": v})
            op.text = "${" + v + "}"

    inner_end = ET.SubElement(wrapper, _bpmn("endEvent"), attrib={"id": inner_end_id})
    ET.SubElement(inner_end, _bpmn("incoming")).text = f"Flow_{wrapper_id}_out"

    ET.SubElement(
        wrapper,
        _bpmn("sequenceFlow"),
        attrib={
            "id": f"Flow_{wrapper_id}_in",
            "sourceRef": inner_start_id,
            "targetRef": plugin_task_id,
        },
    )
    ET.SubElement(
        wrapper,
        _bpmn("sequenceFlow"),
        attrib={
            "id": f"Flow_{wrapper_id}_out",
            "sourceRef": plugin_task_id,
            "targetRef": inner_end_id,
        },
    )

    return wrapper, inputs, outputs


def _build_main_process(
    original_process: ET.Element,
    nodes: Dict[str, Node],
    flows: Dict[str, Flow],
    order: List[str],
    regions: List[Region],
    new_process_id: str,
) -> Tuple[ET.Element, Dict[str, str]]:
    main = ET.Element(original_process.tag, attrib=dict(original_process.attrib))
    main.set("id", new_process_id)
    for child in list(main):
        main.remove(child)

    region_by_node: Dict[str, Region] = {}
    region_first_node: Dict[str, Region] = {}
    for r in regions:
        for nid in r.node_ids:
            region_by_node[nid] = r
        region_first_node[r.node_ids[0]] = r

    for child in list(original_process):
        if _localname(child.tag) == "laneSet":
            main.append(copy.deepcopy(child))

    main_order: List[str] = []
    wrapper_elements: Dict[str, ET.Element] = {}
    wrapper_inputs: Dict[str, List[str]] = {}
    wrapper_outputs: Dict[str, List[str]] = {}

    original_pid_for_topic = original_process.get("id") or "workflow"
    for nid in order:
        if nid in region_by_node:
            r = region_by_node[nid]
            if nid == r.node_ids[0]:
                wrapper, inputs, outputs = _make_wrapper_adhoc(
                    r, nodes, flows, regions, original_pid_for_topic
                )
                wrapper_elements[wrapper.get("id")] = wrapper
                wrapper_inputs[wrapper.get("id")] = inputs
                wrapper_outputs[wrapper.get("id")] = outputs
                main_order.append(wrapper.get("id"))
            continue
        main_order.append(nid)

    wrapper_id_for_region_idx: Dict[int, str] = {}
    for wid, w in wrapper_elements.items():
        fref = w.get(_qhana("fragmentRef"))
        if fref and fref.startswith("E"):
            wrapper_id_for_region_idx[int(fref[1:])] = wid

    wrapper_for_node: Dict[str, str] = {}
    for r in regions:
        wid = wrapper_id_for_region_idx[r.index]
        wrapper_for_node[r.node_ids[0]] = wid
        wrapper_for_node[r.node_ids[-1]] = wid

    interior_of_region: Set[str] = set()
    for r in regions:
        for nid in r.node_ids[1:-1]:
            interior_of_region.add(nid)
        interior_of_region.add(r.node_ids[0])
        interior_of_region.add(r.node_ids[-1])

    surviving_flows: List[ET.Element] = []

    copied_nodes: Dict[str, ET.Element] = {}

    for nid in order:
        if nid in interior_of_region:
            continue
        n = nodes[nid]
        c = copy.deepcopy(n.elem)
        _strip_incoming_outgoing(c)
        copied_nodes[nid] = c

    inserted_wrappers: Set[str] = set()
    for nid in order:
        if nid in region_by_node:
            r = region_by_node[nid]
            if nid == r.node_ids[0]:
                wid = wrapper_for_node[nid]
                if wid not in inserted_wrappers:
                    main.append(wrapper_elements[wid])
                    inserted_wrappers.add(wid)
            continue
        main.append(copied_nodes[nid])

    for f in flows.values():
        src, tgt = f.source, f.target
        skip = False
        for r in regions:
            if f.id in r.internal_flow_ids:
                skip = True
                break
        if skip:
            continue

        new_src = src
        new_tgt = tgt
        for r in regions:
            if src == r.node_ids[-1] and f.id in r.exit_flow_ids:
                new_src = wrapper_id_for_region_idx[r.index]
                break
        for r in regions:
            if tgt == r.node_ids[0] and f.id in r.entry_flow_ids:
                new_tgt = wrapper_id_for_region_idx[r.index]
                break

        if new_src in interior_of_region or new_tgt in interior_of_region:
            continue

        nf = copy.deepcopy(f.elem)
        nf.set("sourceRef", new_src)
        nf.set("targetRef", new_tgt)
        surviving_flows.append(nf)

    for nf in surviving_flows:
        main.append(nf)
        fid = nf.get("id")
        src = nf.get("sourceRef")
        tgt = nf.get("targetRef")
        src_elem = _find_child_by_id(main, src)
        tgt_elem = _find_child_by_id(main, tgt)
        if src_elem is not None:
            _add_outgoing(src_elem, fid)
        if tgt_elem is not None:
            _add_incoming(tgt_elem, fid)

    old_to_new_flow_id: Dict[str, str] = {}
    for f in flows.values():
        internal = False
        for r in regions:
            if f.id in r.internal_flow_ids:
                internal = True
                break
        if not internal:
            old_to_new_flow_id[f.id] = f.id

    for gw in list(main):
        if _localname(gw.tag) != "exclusiveGateway":
            continue
        default = gw.get("default")
        if not default:
            continue
        if default not in old_to_new_flow_id:
            gw.attrib.pop("default", None)

    surviving_ids = {e.get("id") for e in main.iter() if e.get("id")}
    for child in list(original_process):
        local = _localname(child.tag)
        if local in {
            "dataObject",
            "dataObjectReference",
            "dataStoreReference",
            "textAnnotation",
            "group",
        }:
            main.append(copy.deepcopy(child))
            aid = child.get("id")
            if aid:
                surviving_ids.add(aid)

    def remap_for_assoc(ref: str) -> str:
        for r in regions:
            if ref in r.node_ids:
                return wrapper_id_for_region_idx[r.index]
        return ref

    for child in list(original_process):
        if _localname(child.tag) == "association":
            src = remap_for_assoc(child.get("sourceRef") or "")
            tgt = remap_for_assoc(child.get("targetRef") or "")
            if src in surviving_ids and tgt in surviving_ids:
                a = copy.deepcopy(child)
                a.set("sourceRef", src)
                a.set("targetRef", tgt)
                main.append(a)

    for ls in main.findall(f"{{{BPMN_NS}}}laneSet"):
        for lane in ls.findall(f"{{{BPMN_NS}}}lane"):
            old_refs = [
                (r.text or "").strip() for r in lane.findall(f"{{{BPMN_NS}}}flowNodeRef")
            ]
            for r in lane.findall(f"{{{BPMN_NS}}}flowNodeRef"):
                lane.remove(r)
            seen = set()
            for ref in old_refs:
                mapped = ref
                for rg in regions:
                    if ref in rg.node_ids:
                        mapped = wrapper_id_for_region_idx[rg.index]
                        break
                if mapped in surviving_ids and mapped not in seen:
                    e = ET.SubElement(lane, _bpmn("flowNodeRef"))
                    e.text = mapped
                    seen.add(mapped)

    return main, wrapper_for_node


def _find_child_by_id(parent: ET.Element, target_id: str) -> Optional[ET.Element]:
    for child in parent:
        if child.get("id") == target_id:
            return child
    return None


def _build_fragment_process(
    region: Region,
    nodes: Dict[str, Node],
    flows: Dict[str, Flow],
    fragment_process_id: str,
) -> ET.Element:
    fid = f"E{region.index}"
    proc = ET.Element(
        _bpmn("process"),
        attrib={
            "id": fragment_process_id,
            "name": f"Extracted Fragment {fid}",
            "isExecutable": "true",
        },
    )

    start_elem = ET.SubElement(
        proc,
        _bpmn("startEvent"),
        attrib={
            "id": f"StartEvent_{fid}",
        },
    )
    task_elems: List[ET.Element] = []
    for nid in region.node_ids:
        c = copy.deepcopy(nodes[nid].elem)
        _strip_incoming_outgoing(c)
        proc.append(c)
        task_elems.append(c)
    end_elem = ET.SubElement(
        proc,
        _bpmn("endEvent"),
        attrib={
            "id": f"EndEvent_{fid}",
        },
    )

    first_flow_id = f"Flow_{fid}_start"
    sf = ET.SubElement(
        proc,
        _bpmn("sequenceFlow"),
        attrib={
            "id": first_flow_id,
            "sourceRef": f"StartEvent_{fid}",
            "targetRef": region.node_ids[0],
        },
    )
    _add_outgoing(start_elem, first_flow_id)
    _add_incoming(task_elems[0], first_flow_id)

    for i, orig_fid in enumerate(region.internal_flow_ids):
        orig = flows[orig_fid]
        nf = copy.deepcopy(orig.elem)
        nf.set("sourceRef", region.node_ids[i])
        nf.set("targetRef", region.node_ids[i + 1])
        proc.append(nf)
        _add_outgoing(task_elems[i], orig_fid)
        _add_incoming(task_elems[i + 1], orig_fid)

    last_flow_id = f"Flow_{fid}_end"
    sf_end = ET.SubElement(
        proc,
        _bpmn("sequenceFlow"),
        attrib={
            "id": last_flow_id,
            "sourceRef": region.node_ids[-1],
            "targetRef": f"EndEvent_{fid}",
        },
    )
    _add_outgoing(task_elems[-1], last_flow_id)
    _add_incoming(end_elem, last_flow_id)

    return proc


def _new_definitions(original_root: ET.Element) -> ET.Element:
    new_root = ET.Element(original_root.tag, attrib=dict(original_root.attrib))
    return new_root


def _build_main_definitions(
    original_root: ET.Element,
    main_process: ET.Element,
    original_process_id: str,
    new_process_id: str,
    regions: List["Region"],
) -> ET.Element:
    new_root = _new_definitions(original_root)

    for collab in original_root.findall(f"{{{BPMN_NS}}}collaboration"):
        c = copy.deepcopy(collab)
        for p in c.findall(f"{{{BPMN_NS}}}participant"):
            if p.get("processRef") == original_process_id:
                p.set("processRef", new_process_id)
        new_root.append(c)

    for m in original_root.findall(f"{{{BPMN_NS}}}message"):
        new_root.append(copy.deepcopy(m))

    new_root.append(main_process)

    di = _build_main_di(
        original_root, main_process, regions, original_process_id, new_process_id
    )
    if di is not None:
        new_root.append(di)

    return new_root


def _index_original_di(
    original_root: ET.Element,
) -> Tuple[Dict[str, ET.Element], Dict[str, ET.Element], Optional[ET.Element]]:
    shapes: Dict[str, ET.Element] = {}
    edges: Dict[str, ET.Element] = {}
    plane = None
    for diagram in original_root.findall(f"{{{BPMNDI_NS}}}BPMNDiagram"):
        for p in diagram.findall(f"{{{BPMNDI_NS}}}BPMNPlane"):
            if plane is None:
                plane = p
            for el in p.iter():
                local = _localname(el.tag)
                if local == "BPMNShape":
                    ref = el.get("bpmnElement")
                    if ref:
                        shapes[ref] = el
                elif local == "BPMNEdge":
                    ref = el.get("bpmnElement")
                    if ref:
                        edges[ref] = el
    return shapes, edges, plane


def _bounds_of(shape: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    for ch in shape:
        if _localname(ch.tag) == "Bounds":
            try:
                return (
                    float(ch.get("x") or 0),
                    float(ch.get("y") or 0),
                    float(ch.get("width") or 0),
                    float(ch.get("height") or 0),
                )
            except ValueError:
                return None
    return None


def _region_bounding_box(
    region: "Region",
    original_shapes: Dict[str, ET.Element],
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    x2s: List[float] = []
    y2s: List[float] = []
    for nid in region.node_ids:
        shape = original_shapes.get(nid)
        if shape is None:
            continue
        b = _bounds_of(shape)
        if b is None:
            continue
        x, y, w, h = b
        xs.append(x)
        ys.append(y)
        x2s.append(x + w)
        y2s.append(y + h)
    if not xs:
        return (100.0, 100.0, 200.0, 100.0)
    pad = 10.0
    x = min(xs) - pad
    y = min(ys) - pad
    w = max(x2s) - x + pad
    h = max(y2s) - y + pad
    return (x, y, w, h)


def _shape_center(shape: ET.Element) -> Optional[Tuple[float, float]]:
    b = _bounds_of(shape)
    if b is None:
        return None
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def _dock_point(
    shape: ET.Element, toward: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    b = _bounds_of(shape)
    if b is None:
        return None
    x, y, w, h = b
    cx = x + w / 2.0
    cy = y + h / 2.0
    tx, ty = toward
    dx = tx - cx
    dy = ty - cy
    if dx == 0 and dy == 0:
        return (cx, cy)
    hw = w / 2.0
    hh = h / 2.0
    if dx == 0:
        return (cx, cy + (hh if dy > 0 else -hh))
    if dy == 0:
        return (cx + (hw if dx > 0 else -hw), cy)
    sx = hw / abs(dx)
    sy = hh / abs(dy)
    s = min(sx, sy)
    return (cx + dx * s, cy + dy * s)


def _docked_waypoints(
    src_shape: ET.Element,
    tgt_shape: ET.Element,
) -> Optional[List[Tuple[float, float]]]:
    src_center = _shape_center(src_shape)
    tgt_center = _shape_center(tgt_shape)
    if src_center is None or tgt_center is None:
        return None
    src_dock = _dock_point(src_shape, tgt_center)
    tgt_dock = _dock_point(tgt_shape, src_center)
    if src_dock is None or tgt_dock is None:
        return None
    sx, sy = src_dock
    tx, ty = tgt_dock
    if abs(sy - ty) < 2.0 or abs(sx - tx) < 2.0:
        return [src_dock, tgt_dock]
    mid_x = (sx + tx) / 2.0
    return [src_dock, (mid_x, sy), (mid_x, ty), tgt_dock]


def _make_shape(
    bpmn_element_id: str,
    x: float,
    y: float,
    w: float,
    h: float,
    is_expanded: bool = False,
) -> ET.Element:
    shape = ET.Element(
        f"{{{BPMNDI_NS}}}BPMNShape",
        attrib={
            "id": f"{bpmn_element_id}_di",
            "bpmnElement": bpmn_element_id,
        },
    )
    if is_expanded:
        shape.set("isExpanded", "true")
    ET.SubElement(
        shape,
        f"{{{OMGDC_NS}}}Bounds",
        attrib={
            "x": str(int(x)),
            "y": str(int(y)),
            "width": str(int(w)),
            "height": str(int(h)),
        },
    )
    return shape


def _make_edge(flow_id: str, waypoints: List[Tuple[float, float]]) -> ET.Element:
    edge = ET.Element(
        f"{{{BPMNDI_NS}}}BPMNEdge",
        attrib={
            "id": f"{flow_id}_di",
            "bpmnElement": flow_id,
        },
    )
    for x, y in waypoints:
        ET.SubElement(
            edge,
            f"{{{OMGDI_NS}}}waypoint",
            attrib={
                "x": str(int(x)),
                "y": str(int(y)),
            },
        )
    return edge


def _build_main_di(
    original_root: ET.Element,
    main_process: ET.Element,
    regions: List["Region"],
    original_process_id: str,
    new_process_id: str,
) -> Optional[ET.Element]:
    original_shapes, original_edges, _ = _index_original_di(original_root)

    main_direct_ids: Set[str] = set()
    for child in main_process:
        cid = child.get("id")
        if cid:
            main_direct_ids.add(cid)
        for desc in child.iter():
            did = desc.get("id")
            if did:
                main_direct_ids.add(did)
    for child in main_process.findall(f"{{{BPMN_NS}}}sequenceFlow"):
        fid = child.get("id")
        if fid:
            main_direct_ids.add(fid)

    wrappers_by_region_index: Dict[int, str] = {}
    for wchild in main_process.findall(f"{{{BPMN_NS}}}adHocSubProcess"):
        fref = wchild.get(f"{{{QHANA_NS}}}fragmentRef")
        if fref and fref.startswith("E"):
            try:
                idx = int(fref[1:])
                wrappers_by_region_index[idx] = wchild.get("id") or ""
            except ValueError:
                pass

    wrapper_bounds: Dict[str, Tuple[float, float, float, float]] = {}
    for r in regions:
        wid = wrappers_by_region_index.get(r.index)
        if wid:
            wrapper_bounds[wid] = _region_bounding_box(r, original_shapes)

    shapes_by_id: Dict[str, ET.Element] = {}
    for nid, sh in original_shapes.items():
        shapes_by_id[nid] = sh

    diagram = ET.Element(
        f"{{{BPMNDI_NS}}}BPMNDiagram",
        attrib={
            "id": f"BPMNDiagram_{new_process_id}",
        },
    )
    plane = ET.SubElement(
        diagram,
        f"{{{BPMNDI_NS}}}BPMNPlane",
        attrib={
            "id": f"BPMNPlane_{new_process_id}",
            "bpmnElement": new_process_id,
        },
    )

    emitted_shapes: Set[str] = set()

    def emit_shape(cid: str, elem: Optional[ET.Element] = None) -> None:
        if cid in emitted_shapes:
            return
        if cid in wrapper_bounds:
            x, y, w, h = wrapper_bounds[cid]
            shape = _make_shape(cid, x, y, w, h, is_expanded=True)
            plane.append(shape)
            shapes_by_id[cid] = shape
            emitted_shapes.add(cid)
            return
        orig = original_shapes.get(cid)
        if orig is not None:
            plane.append(copy.deepcopy(orig))
            emitted_shapes.add(cid)
            return
        shape = _make_shape(cid, 100, 100, 100, 60)
        plane.append(shape)
        shapes_by_id[cid] = shape
        emitted_shapes.add(cid)

    for child in main_process:
        local = _localname(child.tag)
        if local in (
            "sequenceFlow",
            "laneSet",
            "association",
            "dataObject",
            "dataObjectReference",
            "dataStoreReference",
            "textAnnotation",
            "group",
        ):
            continue
        cid = child.get("id")
        if not cid:
            continue
        emit_shape(cid)
        if local == "adHocSubProcess" and cid not in wrapper_bounds:
            for desc in child.iter():
                if desc is child:
                    continue
                did = desc.get("id")
                dlocal = _localname(desc.tag)
                if (
                    did
                    and dlocal in TASK_LOCALNAMES
                    or desc.tag == _qhana("qHAnaServiceTask")
                ):
                    if did:
                        emit_shape(did)
                elif did and dlocal in ("startEvent", "endEvent"):
                    emit_shape(did)

    for child in main_process:
        local = _localname(child.tag)
        if local in (
            "dataObject",
            "dataObjectReference",
            "dataStoreReference",
            "textAnnotation",
            "group",
        ):
            cid = child.get("id")
            if cid:
                emit_shape(cid)

    for flow_child in main_process.findall(f"{{{BPMN_NS}}}sequenceFlow"):
        fid = flow_child.get("id")
        if not fid:
            continue
        src = flow_child.get("sourceRef") or ""
        tgt = flow_child.get("targetRef") or ""
        orig_edge = original_edges.get(fid)

        synthesize = False
        if orig_edge is None:
            synthesize = True
        else:
            if src in wrapper_bounds or tgt in wrapper_bounds:
                synthesize = True

        if not synthesize and orig_edge is not None:
            plane.append(copy.deepcopy(orig_edge))
            continue

        src_shape = shapes_by_id.get(src)
        if src_shape is None:
            src_shape = original_shapes.get(src)
        tgt_shape = shapes_by_id.get(tgt)
        if tgt_shape is None:
            tgt_shape = original_shapes.get(tgt)
        if src_shape is None or tgt_shape is None:
            plane.append(
                ET.Element(
                    f"{{{BPMNDI_NS}}}BPMNEdge",
                    attrib={
                        "id": f"{fid}_di",
                        "bpmnElement": fid,
                    },
                )
            )
            continue
        waypoints = _docked_waypoints(src_shape, tgt_shape)
        if waypoints is None:
            plane.append(
                ET.Element(
                    f"{{{BPMNDI_NS}}}BPMNEdge",
                    attrib={
                        "id": f"{fid}_di",
                        "bpmnElement": fid,
                    },
                )
            )
            continue
        plane.append(_make_edge(fid, waypoints))

    return diagram


def _build_fragment_di(
    original_root: ET.Element,
    fragment_process: ET.Element,
    region: "Region",
) -> ET.Element:
    original_shapes, original_edges, _ = _index_original_di(original_root)

    fid_label = f"E{region.index}"
    frag_pid = fragment_process.get("id") or ""

    diagram = ET.Element(
        f"{{{BPMNDI_NS}}}BPMNDiagram",
        attrib={
            "id": f"BPMNDiagram_{frag_pid}",
        },
    )
    plane = ET.SubElement(
        diagram,
        f"{{{BPMNDI_NS}}}BPMNPlane",
        attrib={
            "id": f"BPMNPlane_{frag_pid}",
            "bpmnElement": frag_pid,
        },
    )

    bbox = _region_bounding_box(region, original_shapes)
    min_x, min_y, w, h = bbox
    pad = 10.0
    tasks_min_x = min_x + pad
    tasks_max_x = min_x + w - pad
    tasks_mid_y = min_y + h / 2.0

    task_shape_refs: List[ET.Element] = []
    for nid in region.node_ids:
        orig = original_shapes.get(nid)
        if orig is not None:
            copied = copy.deepcopy(orig)
            plane.append(copied)
            task_shape_refs.append(copied)
        else:
            idx = region.node_ids.index(nid)
            fallback = _make_shape(
                nid, tasks_min_x + idx * 140, tasks_mid_y - 40, 100, 80
            )
            plane.append(fallback)
            task_shape_refs.append(fallback)

    first_task = task_shape_refs[0]
    last_task = task_shape_refs[-1]
    first_bounds = _bounds_of(first_task) or (tasks_min_x, tasks_mid_y - 40, 100, 80)
    last_bounds = _bounds_of(last_task) or (tasks_max_x - 100, tasks_mid_y - 40, 100, 80)
    fx, fy, fw, fh = first_bounds
    lx, ly, lw, lh = last_bounds

    start_id = f"StartEvent_{fid_label}"
    start_x = fx - 80
    start_y = fy + fh / 2.0 - 18
    start_shape = _make_shape(start_id, start_x, start_y, 36, 36)
    plane.append(start_shape)

    end_id = f"EndEvent_{fid_label}"
    end_x = lx + lw + 50
    end_y = ly + lh / 2.0 - 18
    end_shape = _make_shape(end_id, end_x, end_y, 36, 36)
    plane.append(end_shape)

    for orig_fid in region.internal_flow_ids:
        orig_edge = original_edges.get(orig_fid)
        if orig_edge is not None:
            plane.append(copy.deepcopy(orig_edge))

    start_to_first_wp = _docked_waypoints(start_shape, first_task)
    last_to_end_wp = _docked_waypoints(last_task, end_shape)
    if start_to_first_wp is not None:
        plane.append(_make_edge(f"Flow_{fid_label}_start", start_to_first_wp))
    if last_to_end_wp is not None:
        plane.append(_make_edge(f"Flow_{fid_label}_end", last_to_end_wp))

    return diagram


def _build_fragment_definitions(
    original_root: ET.Element,
    fragment_process: ET.Element,
    region: "Region",
) -> ET.Element:
    new_root = _new_definitions(original_root)
    new_root.append(fragment_process)
    di = _build_fragment_di(original_root, fragment_process, region)
    new_root.append(di)
    return new_root


def _collect_ids(elem: ET.Element) -> Set[str]:
    ids: Set[str] = set()
    for e in elem.iter():
        i = e.get("id")
        if i:
            ids.add(i)
    return ids


def _serialize(root: ET.Element) -> str:
    """Serialize an ElementTree to an XML string with a declaration."""
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


for _prefix, _uri in (
    ("bpmn", BPMN_NS),
    ("bpmndi", BPMNDI_NS),
    ("omgdc", OMGDC_NS),
    ("omgdi", OMGDI_NS),
    ("camunda", CAMUNDA_NS),
    ("qhana", QHANA_NS),
    ("xsi", XSI_NS),
):
    ET.register_namespace(_prefix, _uri)
del _prefix, _uri


def split_workflow(
    bpmn_xml: str,
    process_id: Optional[str] = None,
    classifier: Optional[Callable[[ET.Element], bool]] = None,
) -> SplitResult:

    classifier = classifier or default_classifier

    original_root = ET.fromstring(bpmn_xml)
    original_process = _find_process(original_root, process_id)
    original_pid = original_process.get("id") or ""

    nodes, flows, order, start_id, end_ids = _parse_process(original_process)
    _classify_nodes(nodes, classifier)
    regions = _find_regions(nodes, flows, order)

    new_main_pid = f"{original_pid}_main"
    main_process, wrapper_for_node = _build_main_process(
        original_process, nodes, flows, order, regions, new_main_pid
    )

    main_defs = _build_main_definitions(
        original_root, main_process, original_pid, new_main_pid, regions
    )
    main_xml = _serialize(main_defs)

    fragments: List[FragmentResult] = []
    for r in regions:
        fid = f"E{r.index}"
        frag_pid = f"{original_pid}_{fid}"
        frag_proc = _build_fragment_process(r, nodes, flows, frag_pid)
        frag_defs = _build_fragment_definitions(original_root, frag_proc, r)
        frag_xml = _serialize(frag_defs)
        inputs, outputs = _compute_region_io(r, nodes, regions)
        fragments.append(
            FragmentResult(
                fragment_id=fid,
                process_id=frag_pid,
                wrapper_id=f"AdHoc_{fid}_Wrapper",
                xml=frag_xml,
                input_variables=inputs,
                output_variables=outputs,
            )
        )

    return SplitResult(main_xml=main_xml, fragments=fragments)


def write_split_outputs(
    bpmn_xml: str,
    out_dir: Path,
    stem: str,
    process_id: Optional[str] = None,
    classifier: Optional[Callable[[ET.Element], bool]] = None,
) -> List[Path]:
    result = split_workflow(bpmn_xml, process_id=process_id, classifier=classifier)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    main_path = out_dir / f"{stem}_main.bpmn"
    main_path.write_text(result.main_xml, encoding="utf-8")
    paths.append(main_path)
    for frag in result.fragments:
        p = out_dir / f"{stem}_{frag.fragment_id}.bpmn"
        p.write_text(frag.xml, encoding="utf-8")
        paths.append(p)
    return paths
