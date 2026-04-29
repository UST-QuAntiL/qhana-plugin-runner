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
    "dc": OMGDC_NS,
    "di": OMGDI_NS,
    "camunda": CAMUNDA_NS,
    "qhana": QHANA_NS,
    "xsi": XSI_NS,
}


class SplitNotSupported(Exception):
    pass


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

    def is_extractable(self, elem: ET.Element) -> bool: ...


def default_classifier(elem: ET.Element) -> bool:
    """Every task-like element is extractable.  The main workflow becomes a
    UI template that cannot execute tasks, so all tasks belong in fragments."""
    tag = elem.tag
    if tag == f"{{{QHANA_NS}}}qHAnaServiceTask":
        return True
    local = _localname(tag)
    return local in TASK_LOCALNAMES


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
    "callActivity",
    "transaction",
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

        if local in {
            "laneSet",
            "documentation",
            "extensionElements",
            "ioSpecification",
        }:
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


def _classify_subprocess(
    elem: ET.Element, classifier: Callable[[ET.Element], bool]
) -> bool:
    inner_classifications: List[Tuple[str, bool]] = []
    for desc in elem.iter():
        if desc is elem:
            continue
        dlocal = _localname(desc.tag)
        if dlocal in TASK_LOCALNAMES or desc.tag == _qhana("qHAnaServiceTask"):
            did = desc.get("id") or "?"
            inner_classifications.append((did, classifier(desc)))
    if not inner_classifications:
        return False
    extractable = [tid for tid, e in inner_classifications if e]
    main_side = [tid for tid, e in inner_classifications if not e]
    if extractable and main_side:
        raise SplitNotSupported(
            f"Mixed subprocess not supported (subprocess id={elem.get('id')!r}): "
            f"extractable tasks {extractable}, main-side tasks {main_side}. "
            "A top-level subprocess must contain only extractable or only "
            "main-side tasks."
        )
    return bool(extractable)


def _is_qhana_task(elem: ET.Element) -> bool:
    if elem.tag == _qhana("qHAnaServiceTask"):
        return True
    local = _localname(elem.tag)
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
    return False


def _validate_adhoc(elem: ET.Element, classifier: Callable[[ET.Element], bool]) -> None:
    aid = elem.get("id") or "?"
    for desc in elem.iter():
        if desc is elem:
            continue
        dlocal = _localname(desc.tag)
        if dlocal == "adHocSubProcess":
            raise SplitNotSupported(
                f"Nested ad-hoc subprocess not supported for UI template generation "
                f"(outer id={aid!r}, inner id={desc.get('id')!r}). "
                f"Use a standard subprocess as the outer container."
            )
        if dlocal in TASK_LOCALNAMES or desc.tag == _qhana("qHAnaServiceTask"):
            if not _is_qhana_task(desc):
                raise SplitNotSupported(
                    f"Ad-hoc subprocess {aid!r} contains non-QHAna task "
                    f"{desc.get('id')!r} (type={dlocal!r}). All tasks inside an "
                    f"ad-hoc must be QHAna-recognized for UI template generation."
                )


def _classify_nodes(
    nodes: Dict[str, Node],
    classifier: Callable[[ET.Element], bool],
    start_id: str,
) -> None:
    for node in nodes.values():
        if node.local in {"exclusiveGateway", "parallelGateway"}:
            node.extractable = False
            continue
        if node.local == "adHocSubProcess":
            _validate_adhoc(node.elem, classifier)
            node.extractable = False
            continue
        if node.local == "subProcess":
            node.extractable = _classify_subprocess(node.elem, classifier)
            continue
        if node.local == "startEvent" and node.id == start_id:
            node.extractable = False
            continue
        if node.local in {
            "startEvent",
            "endEvent",
            "boundaryEvent",
            "intermediateCatchEvent",
            "intermediateThrowEvent",
        }:
            node.extractable = True
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

    def successors(nid: str) -> List[Tuple[str, str]]:
        return [(flows[fid].target, fid) for fid in out_flows.get(nid, [])]

    def predecessors(nid: str) -> List[Tuple[str, str]]:
        return [(flows[fid].source, fid) for fid in in_flows.get(nid, [])]

    def has_extractable_continuation(nid: str, visited: Set[str]) -> bool:
        if nid in visited:
            return False
        visited.add(nid)
        node = nodes.get(nid)
        if node is None:
            return False
        if not node.extractable and nid not in absorbable_gateways:
            return False
        succs = successors(nid)
        if not succs:
            return True
        for target, _ in succs:
            t_node = nodes.get(target)
            if t_node is None:
                continue
            if t_node.extractable or target in absorbable_gateways:
                return True
            if not t_node.extractable:
                continue
        return False

    def reaches_non_absorbed_gateway_upstream(
        start_id: str, absorbable: Set[str]
    ) -> bool:
        visited: Set[str] = set()
        queue = [start_id]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            for source, _ in predecessors(cur):
                s_node = nodes.get(source)
                if s_node is None:
                    continue
                if (
                    s_node.local
                    in {
                        "exclusiveGateway",
                        "parallelGateway",
                    }
                    and source not in absorbable
                ):
                    return True
                if s_node.extractable:
                    queue.append(source)
        return False

    absorbable_gateways: Set[str] = set()
    for nid, node in nodes.items():
        if node.local not in {"exclusiveGateway", "parallelGateway"}:
            continue
        all_ok = True
        for target, _ in successors(nid):
            t_node = nodes.get(target)
            if t_node is None or not t_node.extractable:
                all_ok = False
                break
            t_succs = successors(target)
            if t_succs:
                has_ext_cont = False
                for tt, _ in t_succs:
                    tt_node = nodes.get(tt)
                    if tt_node and (
                        tt_node.extractable
                        or tt_node.local in {"exclusiveGateway", "parallelGateway"}
                    ):
                        has_ext_cont = True
                        break
                if not has_ext_cont:
                    all_ok = False
                    break
        if not all_ok:
            continue
        for source, _ in predecessors(nid):
            s_node = nodes.get(source)
            if s_node is None:
                all_ok = False
                break
            if not s_node.extractable and s_node.local not in {
                "exclusiveGateway",
                "parallelGateway",
                "startEvent",
            }:
                all_ok = False
                break
        if all_ok:
            absorbable_gateways.add(nid)

    changed = True
    while changed:
        changed = False
        for gid in list(absorbable_gateways):
            if reaches_non_absorbed_gateway_upstream(gid, absorbable_gateways):
                absorbable_gateways.discard(gid)
                changed = True
                continue
            keep = True
            for target, _ in successors(gid):
                t_node = nodes.get(target)
                if (
                    t_node
                    and not t_node.extractable
                    and target not in absorbable_gateways
                ):
                    keep = False
                    break
            if not keep:
                absorbable_gateways.discard(gid)
                changed = True

    effectively_extractable: Set[str] = set()
    for nid, node in nodes.items():
        if node.extractable or nid in absorbable_gateways:
            effectively_extractable.add(nid)

    assigned: Set[str] = set()
    regions: List[Region] = []
    idx = 0

    for nid in order:
        if nid in assigned:
            continue
        if nid not in effectively_extractable:
            continue

        component: List[str] = []
        queue = [nid]
        visited: Set[str] = {nid}
        while queue:
            cur = queue.pop(0)
            component.append(cur)
            for target, _ in successors(cur):
                if target not in visited and target in effectively_extractable:
                    visited.add(target)
                    queue.append(target)
            for source, _ in predecessors(cur):
                if source not in visited and source in effectively_extractable:
                    visited.add(source)
                    queue.append(source)

        boundary_added = True
        while boundary_added:
            boundary_added = False
            for bnid, bnode in nodes.items():
                if bnid in visited:
                    continue
                if bnode.local == "boundaryEvent":
                    host = bnode.elem.get("attachedToRef")
                    if host and host in visited:
                        visited.add(bnid)
                        component.append(bnid)
                        boundary_added = True
                        bqueue = [bnid]
                        while bqueue:
                            bc = bqueue.pop(0)
                            for bt, _ in successors(bc):
                                if bt not in visited and bt in effectively_extractable:
                                    visited.add(bt)
                                    component.append(bt)
                                    bqueue.append(bt)

        order_index = {nid: i for i, nid in enumerate(order)}
        for c in component:
            if c not in order_index:
                host = (
                    nodes[c].elem.get("attachedToRef")
                    if nodes[c].local == "boundaryEvent"
                    else None
                )
                order_index[c] = order_index.get(host, 99999) + 0.5
        component.sort(key=lambda x: order_index.get(x, 99999))

        component_set = set(component)

        internal_flow_ids: List[str] = []
        entry_flow_ids: List[str] = []
        exit_flow_ids: List[str] = []

        for fid, flow in flows.items():
            src_in = flow.source in component_set
            tgt_in = flow.target in component_set
            if src_in and tgt_in:
                internal_flow_ids.append(fid)
            elif not src_in and tgt_in:
                entry_flow_ids.append(fid)
            elif src_in and not tgt_in:
                exit_flow_ids.append(fid)

        has_task = any(
            nodes[c].local in TASK_LOCALNAMES
            or nodes[c].elem.tag == _qhana("qHAnaServiceTask")
            or nodes[c].local == "subProcess"
            for c in component
        )
        if not has_task:
            continue

        idx += 1
        regions.append(
            Region(
                index=idx,
                node_ids=component,
                entry_flow_ids=entry_flow_ids,
                exit_flow_ids=exit_flow_ids,
                internal_flow_ids=internal_flow_ids,
            )
        )
        assigned.update(component)

    return regions


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

    has_end = any(_localname(c.tag) == "endEvent" for c in main if c.get("id"))
    if not has_end:
        synth_end_id = "EndEvent_Main"
        synth_end = ET.SubElement(
            main, _bpmn("endEvent"), attrib={"id": synth_end_id, "name": "End"}
        )

        main_elem_ids = {c.get("id") for c in main if c.get("id")}
        targets_of_flows = set()
        sources_of_flows = set()
        for sf in main:
            if _localname(sf.tag) == "sequenceFlow":
                targets_of_flows.add(sf.get("targetRef"))
                sources_of_flows.add(sf.get("sourceRef"))

        leaf_ids = []
        for nid in reversed(main_order):
            eid = nid
            if eid in main_elem_ids and eid in sources_of_flows:
                continue
            if eid in main_elem_ids and eid not in sources_of_flows:
                leaf_ids.append(eid)

        if not leaf_ids:
            if main_order:
                leaf_ids = [main_order[-1]]

        for i, leaf_id in enumerate(leaf_ids):
            flow_id = (
                "Flow_to_Main_End" if len(leaf_ids) == 1 else f"Flow_to_Main_End_{i}"
            )
            ET.SubElement(
                main,
                _bpmn("sequenceFlow"),
                attrib={
                    "id": flow_id,
                    "sourceRef": leaf_id,
                    "targetRef": synth_end_id,
                },
            )
            leaf_elem = _find_child_by_id(main, leaf_id)
            if leaf_elem is not None:
                _add_outgoing(leaf_elem, flow_id)
            _add_incoming(synth_end, flow_id)

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

    seen_visual_assocs = set()
    for r in regions:
        wrapper_id = wrapper_id_for_region_idx[r.index]
        for nid in r.node_ids:
            orig_node = nodes.get(nid)
            if orig_node is None:
                continue
            for desc in orig_node.elem.iter():
                dlocal = _localname(desc.tag)
                if dlocal == "dataOutputAssociation":
                    for tgt_el in desc.findall(f"{{{BPMN_NS}}}targetRef"):
                        ref = (tgt_el.text or "").strip()
                        if ref in surviving_ids:
                            key = (wrapper_id, ref)
                            if key in seen_visual_assocs:
                                continue
                            seen_visual_assocs.add(key)
                            a = ET.SubElement(
                                main,
                                _bpmn("association"),
                                attrib={
                                    "id": f"Assoc_{wrapper_id}_to_{ref}",
                                    "sourceRef": wrapper_id,
                                    "targetRef": ref,
                                    "associationDirection": "One",
                                },
                            )
                elif dlocal == "dataInputAssociation":
                    for src_el in desc.findall(f"{{{BPMN_NS}}}sourceRef"):
                        ref = (src_el.text or "").strip()
                        if ref in surviving_ids:
                            key = (ref, wrapper_id)
                            if key in seen_visual_assocs:
                                continue
                            seen_visual_assocs.add(key)
                            a = ET.SubElement(
                                main,
                                _bpmn("association"),
                                attrib={
                                    "id": f"Assoc_{ref}_to_{wrapper_id}",
                                    "sourceRef": ref,
                                    "targetRef": wrapper_id,
                                    "associationDirection": "One",
                                },
                            )

    for block in main.iter():
        blocal = _localname(block.tag)
        if blocal not in ("adHocSubProcess", "subProcess"):
            continue
        block_id = block.get("id")
        if not block_id or block_id in wrapper_id_for_region_idx.values():
            continue  # skip our synthesized wrappers
        for desc in block.iter():
            if desc is block:
                continue
            dlocal = _localname(desc.tag)
            if dlocal == "dataOutputAssociation":
                for tgt_el in desc.findall(f"{{{BPMN_NS}}}targetRef"):
                    ref = (tgt_el.text or "").strip()
                    if ref in surviving_ids:
                        key = (block_id, ref)
                        if key in seen_visual_assocs:
                            continue
                        seen_visual_assocs.add(key)
                        ET.SubElement(
                            main,
                            _bpmn("association"),
                            attrib={
                                "id": f"Assoc_{block_id}_to_{ref}",
                                "sourceRef": block_id,
                                "targetRef": ref,
                                "associationDirection": "One",
                            },
                        )
            elif dlocal == "dataInputAssociation":
                for src_el in desc.findall(f"{{{BPMN_NS}}}sourceRef"):
                    ref = (src_el.text or "").strip()
                    if ref in surviving_ids:
                        key = (ref, block_id)
                        if key in seen_visual_assocs:
                            continue
                        seen_visual_assocs.add(key)
                        ET.SubElement(
                            main,
                            _bpmn("association"),
                            attrib={
                                "id": f"Assoc_{ref}_to_{block_id}",
                                "sourceRef": ref,
                                "targetRef": block_id,
                                "associationDirection": "One",
                            },
                        )

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

    component_set = set(region.node_ids)

    internal_flow_sources: Set[str] = set()
    internal_flow_targets: Set[str] = set()
    for fid_flow in region.internal_flow_ids:
        f = flows[fid_flow]
        internal_flow_sources.add(f.source)
        internal_flow_targets.add(f.target)

    boundary_hosts_in_region: Set[str] = set()
    for nid in region.node_ids:
        node = nodes[nid]
        if node.local == "boundaryEvent":
            host = node.elem.get("attachedToRef")
            if host and host in component_set:
                boundary_hosts_in_region.add(host)

    entry_nodes: Set[str] = set()
    for nid in region.node_ids:
        if nid in internal_flow_targets:
            continue
        if nodes[nid].local == "boundaryEvent":
            continue
        entry_nodes.add(nid)

    if not entry_nodes:
        for efid in region.entry_flow_ids:
            target = flows[efid].target
            entry_nodes.add(target)
            break

    has_original_start = any(nodes[nid].local == "startEvent" for nid in region.node_ids)
    has_original_end = any(nodes[nid].local == "endEvent" for nid in region.node_ids)

    exit_nodes: Set[str] = set()
    for efid in region.exit_flow_ids:
        exit_nodes.add(flows[efid].source)
    for nid in region.node_ids:
        if nid not in internal_flow_sources and nid not in boundary_hosts_in_region:
            if nodes[nid].local != "boundaryEvent":
                exit_nodes.add(nid)

    need_synth_start = not has_original_start
    need_synth_end = not has_original_end

    synth_start_id = f"StartEvent_{fid}"
    synth_end_id = f"EndEvent_{fid}"

    if need_synth_start:
        start_elem = ET.SubElement(
            proc, _bpmn("startEvent"), attrib={"id": synth_start_id, "name": "Start"}
        )

    elem_by_id: Dict[str, ET.Element] = {}
    for nid in region.node_ids:
        c = copy.deepcopy(nodes[nid].elem)
        _strip_incoming_outgoing(c)
        proc.append(c)
        elem_by_id[nid] = c

    if need_synth_end:
        end_elem = ET.SubElement(
            proc, _bpmn("endEvent"), attrib={"id": synth_end_id, "name": "End"}
        )

    if need_synth_start:
        if len(entry_nodes) == 1:
            entry_nid = next(iter(entry_nodes))
            flow_id = f"Flow_{fid}_start"
            ET.SubElement(
                proc,
                _bpmn("sequenceFlow"),
                attrib={
                    "id": flow_id,
                    "sourceRef": synth_start_id,
                    "targetRef": entry_nid,
                },
            )
            _add_outgoing(start_elem, flow_id)
            if entry_nid in elem_by_id:
                _add_incoming(elem_by_id[entry_nid], flow_id)
        elif len(entry_nodes) > 1:
            synth_fork_id = f"Gateway_{fid}_Fork"
            fork_elem = ET.SubElement(
                proc,
                _bpmn("parallelGateway"),
                attrib={"id": synth_fork_id},
            )
            fork_flow_id = f"Flow_{fid}_to_fork"
            ET.SubElement(
                proc,
                _bpmn("sequenceFlow"),
                attrib={
                    "id": fork_flow_id,
                    "sourceRef": synth_start_id,
                    "targetRef": synth_fork_id,
                },
            )
            _add_outgoing(start_elem, fork_flow_id)
            _add_incoming(fork_elem, fork_flow_id)
            for i, entry_nid in enumerate(sorted(entry_nodes)):
                flow_id = f"Flow_{fid}_fork_{i}"
                ET.SubElement(
                    proc,
                    _bpmn("sequenceFlow"),
                    attrib={
                        "id": flow_id,
                        "sourceRef": synth_fork_id,
                        "targetRef": entry_nid,
                    },
                )
                _add_outgoing(fork_elem, flow_id)
                if entry_nid in elem_by_id:
                    _add_incoming(elem_by_id[entry_nid], flow_id)

    for ifid in region.internal_flow_ids:
        orig = flows[ifid]
        nf = copy.deepcopy(orig.elem)
        proc.append(nf)
        if orig.source in elem_by_id:
            _add_outgoing(elem_by_id[orig.source], ifid)
        if orig.target in elem_by_id:
            _add_incoming(elem_by_id[orig.target], ifid)

    if need_synth_end:
        wire_to_end: Set[str] = set()
        for nid in exit_nodes:
            if nodes[nid].local == "endEvent":
                continue
            if nid not in internal_flow_sources:
                wire_to_end.add(nid)
        for i, exit_nid in enumerate(sorted(wire_to_end)):
            flow_id = (
                f"Flow_{fid}_end" if len(wire_to_end) == 1 else f"Flow_{fid}_end_{i}"
            )
            ET.SubElement(
                proc,
                _bpmn("sequenceFlow"),
                attrib={
                    "id": flow_id,
                    "sourceRef": exit_nid,
                    "targetRef": synth_end_id,
                },
            )
            if exit_nid in elem_by_id:
                _add_outgoing(elem_by_id[exit_nid], flow_id)
            _add_incoming(end_elem, flow_id)

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


def _find_shape_in_plane(plane: ET.Element, element_id: str) -> Optional[ET.Element]:
    for child in plane:
        if (
            _localname(child.tag) == "BPMNShape"
            and child.get("bpmnElement") == element_id
        ):
            return child
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
        if cid == "EndEvent_Main":
            continue
        emit_shape(cid)
        if local in ("adHocSubProcess", "subProcess") and cid not in wrapper_bounds:
            emitted_ids_local = set(emitted_shapes)
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
                elif did and dlocal in (
                    "startEvent",
                    "endEvent",
                    "adHocSubProcess",
                    "subProcess",
                    "boundaryEvent",
                    "intermediateCatchEvent",
                    "intermediateThrowEvent",
                ):
                    emit_shape(did)
                elif did and dlocal == "sequenceFlow":
                    orig_edge = original_edges.get(did)
                    if orig_edge is not None:
                        edge_copy = copy.deepcopy(orig_edge)
                        already = any(
                            _localname(e.tag) == "BPMNEdge"
                            and e.get("bpmnElement") == did
                            for e in plane
                        )
                        if not already:
                            plane.append(edge_copy)

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
        if fid.startswith("Flow_to_Main_End"):
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

    for assoc in main_process.findall(f"{{{BPMN_NS}}}association"):
        aid = assoc.get("id") or ""
        if not aid.startswith("Assoc_"):
            continue
        if aid in original_edges:
            continue
        src = assoc.get("sourceRef") or ""
        tgt = assoc.get("targetRef") or ""
        src_shape = shapes_by_id.get(src)
        if src_shape is None:
            src_shape = original_shapes.get(src)
        tgt_shape = shapes_by_id.get(tgt)
        if tgt_shape is None:
            tgt_shape = original_shapes.get(tgt)
        if src_shape is None or tgt_shape is None:
            continue
        waypoints = _docked_waypoints(src_shape, tgt_shape)
        if waypoints is None:
            continue
        plane.append(_make_edge(aid, waypoints))

    main_ids = _collect_ids(main_process)
    if "EndEvent_Main" in main_ids:
        max_x = 0.0
        rightmost_shape = None
        for s in plane:
            if _localname(s.tag) != "BPMNShape":
                continue
            b = _bounds_of(s)
            if b is not None and (b[0] + b[2]) > max_x:
                max_x = b[0] + b[2]
                rightmost_shape = s

        end_x = max_x + 60
        end_y = 200.0
        if rightmost_shape is not None:
            rb = _bounds_of(rightmost_shape)
            if rb:
                end_y = rb[1] + rb[3] / 2.0 - 18

        end_shape = _make_shape("EndEvent_Main", end_x, end_y, 36, 36)
        plane.append(end_shape)

        for sf in main_process:
            if _localname(sf.tag) != "sequenceFlow":
                continue
            fid = sf.get("id") or ""
            if not fid.startswith("Flow_to_Main_End"):
                continue
            src = sf.get("sourceRef") or ""
            src_shape = shapes_by_id.get(src)
            if src_shape is None:
                src_shape = original_shapes.get(src)
            if src_shape is None:
                continue
            waypoints = _docked_waypoints(src_shape, end_shape)
            if waypoints is not None:
                plane.append(_make_edge(fid, waypoints))

    return diagram


def _collect_ids(elem: ET.Element) -> Set[str]:
    ids: Set[str] = set()
    for e in elem.iter():
        i = e.get("id")
        if i:
            ids.add(i)
    return ids


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

    region_node_elems = {nid: None for nid in region.node_ids}
    for nid in region.node_ids:
        region_node_elems[nid] = None
    emitted_frag_ids = {
        s.get("bpmnElement") for s in plane if _localname(s.tag) == "BPMNShape"
    }
    for nid in region.node_ids:
        node_elem = None
        for n in region_node_elems:
            pass
        for child in region_node_elems:
            pass
    for proc in original_root.iter(f"{{{BPMN_NS}}}process"):
        for child in proc.iter():
            cid = child.get("id")
            if cid not in region.node_ids:
                continue
            clocal = _localname(child.tag)
            if clocal not in ("subProcess", "adHocSubProcess"):
                continue
            for desc in child.iter():
                if desc is child:
                    continue
                did = desc.get("id")
                if not did or did in emitted_frag_ids:
                    continue
                dlocal = _localname(desc.tag)
                if (
                    dlocal in TASK_LOCALNAMES
                    or desc.tag == _qhana("qHAnaServiceTask")
                    or dlocal
                    in (
                        "startEvent",
                        "endEvent",
                        "adHocSubProcess",
                        "subProcess",
                        "boundaryEvent",
                        "intermediateCatchEvent",
                        "intermediateThrowEvent",
                    )
                ):
                    orig = original_shapes.get(did)
                    if orig is not None:
                        plane.append(copy.deepcopy(orig))
                        emitted_frag_ids.add(did)
                elif dlocal == "sequenceFlow":
                    orig_edge = original_edges.get(did)
                    if orig_edge is not None:
                        plane.append(copy.deepcopy(orig_edge))
                        emitted_frag_ids.add(did)

    first_task = task_shape_refs[0]
    last_task = task_shape_refs[-1]
    first_bounds = _bounds_of(first_task) or (tasks_min_x, tasks_mid_y - 40, 100, 80)
    last_bounds = _bounds_of(last_task) or (tasks_max_x - 100, tasks_mid_y - 40, 100, 80)
    fx, fy, fw, fh = first_bounds
    lx, ly, lw, lh = last_bounds

    frag_ids = _collect_ids(fragment_process)

    start_id = f"StartEvent_{fid_label}"
    start_x = fx - 80
    start_y = fy + fh / 2.0 - 18
    if start_id in frag_ids:
        start_shape = _make_shape(start_id, start_x, start_y, 36, 36)
        plane.append(start_shape)
    else:
        start_shape = None

    end_id = f"EndEvent_{fid_label}"
    end_x = lx + lw + 50
    end_y = ly + lh / 2.0 - 18
    if end_id in frag_ids:
        end_shape = _make_shape(end_id, end_x, end_y, 36, 36)
        plane.append(end_shape)
    else:
        end_shape = None

    for orig_fid in region.internal_flow_ids:
        orig_edge = original_edges.get(orig_fid)
        if orig_edge is not None:
            plane.append(copy.deepcopy(orig_edge))

    if start_shape is not None and f"Flow_{fid_label}_start" in frag_ids:
        start_to_first_wp = _docked_waypoints(start_shape, first_task)
        if start_to_first_wp is not None:
            plane.append(_make_edge(f"Flow_{fid_label}_start", start_to_first_wp))
    if end_shape is not None and f"Flow_{fid_label}_end" in frag_ids:
        last_to_end_wp = _docked_waypoints(last_task, end_shape)
        if last_to_end_wp is not None:
            plane.append(_make_edge(f"Flow_{fid_label}_end", last_to_end_wp))

    fork_id = f"Gateway_{fid_label}_Fork"
    if fork_id in frag_ids and start_shape is not None:
        entry_task_shapes = []
        for sf in fragment_process:
            if not sf.tag.endswith("sequenceFlow"):
                continue
            sfid = sf.get("id") or ""
            if not sfid.startswith(f"Flow_{fid_label}_fork_"):
                continue
            tgt = sf.get("targetRef") or ""
            tgt_shape = _find_shape_in_plane(plane, tgt)
            if tgt_shape is None:
                tgt_shape = original_shapes.get(tgt)
            if tgt_shape is not None:
                entry_task_shapes.append(tgt_shape)

        if entry_task_shapes:
            min_entry_x = min(_bounds_of(s)[0] for s in entry_task_shapes)
            avg_entry_y = sum(
                _bounds_of(s)[1] + _bounds_of(s)[3] / 2.0 for s in entry_task_shapes
            ) / len(entry_task_shapes)

            new_start_x = min_entry_x - 180
            new_start_y = avg_entry_y - 18
            new_fork_x = min_entry_x - 80
            new_fork_y = avg_entry_y - 25

            sb = start_shape.find(f"{{{OMGDC_NS}}}Bounds")
            if sb is not None:
                sb.set("x", str(new_start_x))
                sb.set("y", str(new_start_y))

            fork_shape = _make_shape(fork_id, new_fork_x, new_fork_y, 50, 50)
            plane.append(fork_shape)
        else:
            fork_x = start_x + 80
            fork_y = start_y - 4
            fork_shape = _make_shape(fork_id, fork_x, fork_y, 50, 50)
            plane.append(fork_shape)

        to_fork_id = f"Flow_{fid_label}_to_fork"
        if to_fork_id in frag_ids:
            wp = _docked_waypoints(start_shape, fork_shape)
            if wp is not None:
                plane.append(_make_edge(to_fork_id, wp))

        for sf in fragment_process:
            if not sf.tag.endswith("sequenceFlow"):
                continue
            sfid = sf.get("id") or ""
            if not sfid.startswith(f"Flow_{fid_label}_fork_"):
                continue
            tgt = sf.get("targetRef") or ""
            tgt_shape = _find_shape_in_plane(plane, tgt)
            if tgt_shape is None:
                tgt_shape = original_shapes.get(tgt)
            if tgt_shape is None:
                continue
            wp = _docked_waypoints(fork_shape, tgt_shape)
            if wp is not None:
                plane.append(_make_edge(sfid, wp))

    if end_shape is not None:
        for sf in fragment_process:
            if not sf.tag.endswith("sequenceFlow"):
                continue
            sfid = sf.get("id") or ""
            if not sfid.startswith(f"Flow_{fid_label}_end_"):
                continue
            src = sf.get("sourceRef") or ""
            src_shape = _find_shape_in_plane(plane, src)
            if src_shape is None:
                src_shape = original_shapes.get(src)
            if src_shape is None:
                continue
            wp = _docked_waypoints(src_shape, end_shape)
            if wp is not None:
                plane.append(_make_edge(sfid, wp))

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


def _serialize(root: ET.Element) -> str:
    """Serialize an ElementTree to an XML string with a declaration."""
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


for _prefix, _uri in (
    ("bpmn", BPMN_NS),
    ("bpmndi", BPMNDI_NS),
    ("dc", OMGDC_NS),
    ("di", OMGDI_NS),
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
    _classify_nodes(nodes, classifier, start_id)
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
