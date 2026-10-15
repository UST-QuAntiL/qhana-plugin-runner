import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Set, Tuple

from .models import (
    BPMN_NS,
    CAMUNDA_NS,
    QHANA_NS,
    TASK_LOCALNAMES,
    UNSUPPORTED_LOCALNAMES,
    Node,
    Flow,
    Region,
    SplitNotSupported,
    _localname,
    _bpmn,
    _qhana,
    _camunda,
    _is_task_like,
)


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
