import copy
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    BPMN_NS,
    BPMNDI_NS,
    OMGDC_NS,
    OMGDI_NS,
    CAMUNDA_NS,
    QHANA_NS,
    XSI_NS,
    NS,
    TASK_LOCALNAMES,
    Node,
    Flow,
    Region,
    FragmentResult,
    SplitNotSupported,
    _localname,
    _bpmn,
    _qhana,
    _camunda,
    _is_task_like,
    _strip_incoming_outgoing,
    _add_incoming,
    _add_outgoing,
)

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

    data_refs_in_fragments: Set[str] = set()
    for r in regions:
        for nid in r.node_ids:
            node_elem = nodes[nid].elem
            for desc in node_elem.iter():
                dlocal = _localname(desc.tag)
                if dlocal == "dataOutputAssociation":
                    for tgt_el in desc.findall(f"{{{BPMN_NS}}}targetRef"):
                        ref = (tgt_el.text or "").strip()
                        if ref:
                            data_refs_in_fragments.add(ref)
                elif dlocal == "dataInputAssociation":
                    for src_el in desc.findall(f"{{{BPMN_NS}}}sourceRef"):
                        ref = (src_el.text or "").strip()
                        if ref:
                            data_refs_in_fragments.add(ref)

    data_refs_in_main: Set[str] = set()
    main_node_ids = {e.get("id") for e in main.iter() if e.get("id")}
    for child in original_process:
        local = _localname(child.tag)
        cid = child.get("id") or ""
        if cid in main_node_ids:
            for desc in child.iter():
                dlocal = _localname(desc.tag)
                if dlocal == "dataOutputAssociation":
                    for tgt_el in desc.findall(f"{{{BPMN_NS}}}targetRef"):
                        ref = (tgt_el.text or "").strip()
                        if ref:
                            data_refs_in_main.add(ref)
                elif dlocal == "dataInputAssociation":
                    for src_el in desc.findall(f"{{{BPMN_NS}}}sourceRef"):
                        ref = (src_el.text or "").strip()
                        if ref:
                            data_refs_in_main.add(ref)

    surviving_ids = {e.get("id") for e in main.iter() if e.get("id")}
    for child in list(original_process):
        local = _localname(child.tag)
        if local in {
            "dataObject",
            "dataObjectReference",
            "dataStoreReference",
            "group",
        }:
            cid = child.get("id") or ""
            if cid in data_refs_in_fragments and cid not in data_refs_in_main:
                continue
            main.append(copy.deepcopy(child))
            if cid:
                surviving_ids.add(cid)
        elif local == "textAnnotation":
            ann_id = child.get("id") or ""
            all_endpoints_in_fragments = True
            has_any_assoc = False
            for assoc in original_process:
                if _localname(assoc.tag) != "association":
                    continue
                src = assoc.get("sourceRef") or ""
                tgt = assoc.get("targetRef") or ""
                if src == ann_id or tgt == ann_id:
                    has_any_assoc = True
                    other = tgt if src == ann_id else src
                    all_region_ids = set()
                    for r in regions:
                        all_region_ids.update(r.node_ids)
                    if other not in all_region_ids:
                        all_endpoints_in_fragments = False
                        break
            if has_any_assoc and all_endpoints_in_fragments:
                continue
            main.append(copy.deepcopy(child))
            if ann_id:
                surviving_ids.add(ann_id)

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
                            ET.SubElement(
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
                            ET.SubElement(
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
        if not block_id:
            continue
        if block_id in wrapper_id_for_region_idx.values():
            continue
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
    original_process: ET.Element,
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

    data_refs_in_fragment: Set[str] = set()
    for nid in region.node_ids:
        node_elem = nodes[nid].elem
        for desc in node_elem.iter():
            dlocal = _localname(desc.tag)
            if dlocal == "dataOutputAssociation":
                for tgt_el in desc.findall(f"{{{BPMN_NS}}}targetRef"):
                    ref = (tgt_el.text or "").strip()
                    if ref:
                        data_refs_in_fragment.add(ref)
            elif dlocal == "dataInputAssociation":
                for src_el in desc.findall(f"{{{BPMN_NS}}}sourceRef"):
                    ref = (src_el.text or "").strip()
                    if ref:
                        data_refs_in_fragment.add(ref)

    for child in original_process:
        local = _localname(child.tag)
        cid = child.get("id") or ""
        if (
            local
            in {
                "dataObject",
                "dataObjectReference",
                "dataStoreReference",
            }
            and cid in data_refs_in_fragment
        ):
            proc.append(copy.deepcopy(child))
            obj_ref = child.get("dataObjectRef") or child.get("dataStoreRef")
            if obj_ref:
                for sibling in original_process:
                    if (
                        _localname(sibling.tag)
                        in {
                            "dataObject",
                            "dataStore",
                        }
                        and sibling.get("id") == obj_ref
                    ):
                        proc.append(copy.deepcopy(sibling))

    frag_all_ids = {e.get("id") for e in proc.iter() if e.get("id")}
    for child in original_process:
        if _localname(child.tag) != "textAnnotation":
            continue
        ann_id = child.get("id")
        for assoc in original_process:
            if _localname(assoc.tag) != "association":
                continue
            src = assoc.get("sourceRef") or ""
            tgt = assoc.get("targetRef") or ""
            other = None
            if src == ann_id and tgt in component_set:
                other = tgt
            elif tgt == ann_id and src in component_set:
                other = src
            if other is not None:
                proc.append(copy.deepcopy(child))
                a = copy.deepcopy(assoc)
                proc.append(a)
                break

    return proc
