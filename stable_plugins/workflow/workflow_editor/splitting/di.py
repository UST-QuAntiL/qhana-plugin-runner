import copy
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
    Region,
    _localname,
    _bpmn,
    _qhana,
)


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
        src = assoc.get("sourceRef") or ""
        tgt = assoc.get("targetRef") or ""
        orig_edge = original_edges.get(aid)
        if orig_edge is not None:
            src_shape = shapes_by_id.get(src)
            if src_shape is None:
                src_shape = original_shapes.get(src)
            tgt_shape = shapes_by_id.get(tgt)
            if tgt_shape is None:
                tgt_shape = original_shapes.get(tgt)
            if src_shape is not None and tgt_shape is not None:
                if src in wrapper_bounds or tgt in wrapper_bounds:
                    waypoints = _docked_waypoints(src_shape, tgt_shape)
                    if waypoints is not None:
                        plane.append(_make_edge(aid, waypoints))
                else:
                    plane.append(copy.deepcopy(orig_edge))
            else:
                plane.append(copy.deepcopy(orig_edge))
        else:
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

    emitted_edge_ids = {
        e.get("bpmnElement") for e in plane if _localname(e.tag) == "BPMNEdge"
    }
    for child in main_process:
        for desc in child.iter():
            dl = _localname(desc.tag)
            if dl not in (
                "dataOutputAssociation",
                "dataInputAssociation",
            ):
                continue
            did = desc.get("id") or ""
            if did in emitted_edge_ids:
                continue
            orig_edge = original_edges.get(did)
            if orig_edge is not None:
                plane.append(copy.deepcopy(orig_edge))
            else:
                if dl == "dataOutputAssociation":
                    src_id = child.get("id") or ""
                    tgt_el = desc.find(f"{{{BPMN_NS}}}targetRef")
                    tgt_id = (tgt_el.text or "").strip() if tgt_el is not None else ""
                else:
                    tgt_id = child.get("id") or ""
                    src_el = desc.find(f"{{{BPMN_NS}}}sourceRef")
                    src_id = (src_el.text or "").strip() if src_el is not None else ""
                src_shape = shapes_by_id.get(src_id)
                if src_shape is None:
                    src_shape = original_shapes.get(src_id)
                tgt_shape = shapes_by_id.get(tgt_id)
                if tgt_shape is None:
                    tgt_shape = original_shapes.get(tgt_id)
                if src_shape is not None and tgt_shape is not None:
                    waypoints = _docked_waypoints(src_shape, tgt_shape)
                    if waypoints is not None:
                        plane.append(_make_edge(did, waypoints))

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
    for child in fragment_process:
        if _localname(child.tag) == "startEvent":
            start_id = child.get("id") or start_id
            break
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

    frag_proc_ids = _collect_ids(fragment_process)
    data_ann_locals = {
        "dataObject",
        "dataObjectReference",
        "dataStoreReference",
        "textAnnotation",
        "group",
    }
    for child in fragment_process:
        local = _localname(child.tag)
        cid = child.get("id") or ""
        if local in data_ann_locals and cid not in emitted_frag_ids:
            orig = original_shapes.get(cid)
            if orig is not None:
                shape = copy.deepcopy(orig)
                plane.append(shape)
                emitted_frag_ids.add(cid)
            else:
                b = _bounds_of(last_task) if last_task is not None else None
                if b is not None:
                    dx = b[0]
                    dy = b[1] + b[3] + 40
                else:
                    dx, dy = tasks_min_x, tasks_mid_y + 100
                fallback = _make_shape(cid, dx, dy, 50, 50)
                plane.append(fallback)
                emitted_frag_ids.add(cid)

    for child in fragment_process:
        if _localname(child.tag) != "association":
            continue
        aid = child.get("id") or ""
        if aid in emitted_frag_ids:
            continue
        orig_edge = original_edges.get(aid)
        if orig_edge is not None:
            plane.append(copy.deepcopy(orig_edge))
            emitted_frag_ids.add(aid)
        else:
            src = child.get("sourceRef") or ""
            tgt = child.get("targetRef") or ""
            src_shape = _find_shape_in_plane(plane, src)
            if src_shape is None:
                src_shape = original_shapes.get(src)
            tgt_shape = _find_shape_in_plane(plane, tgt)
            if tgt_shape is None:
                tgt_shape = original_shapes.get(tgt)
            if src_shape is not None and tgt_shape is not None:
                wp = _docked_waypoints(src_shape, tgt_shape)
                if wp is not None:
                    plane.append(_make_edge(aid, wp))
                    emitted_frag_ids.add(aid)

    for child in fragment_process:
        for desc in child.iter():
            dl = _localname(desc.tag)
            if dl not in (
                "dataOutputAssociation",
                "dataInputAssociation",
            ):
                continue
            did = desc.get("id") or ""
            if did in emitted_frag_ids:
                continue
            orig_edge = original_edges.get(did)
            if orig_edge is not None:
                plane.append(copy.deepcopy(orig_edge))
                emitted_frag_ids.add(did)
            else:
                if dl == "dataOutputAssociation":
                    src_id = child.get("id") or ""
                    tgt_el = desc.find(f"{{{BPMN_NS}}}targetRef")
                    tgt_id = (tgt_el.text or "").strip() if tgt_el is not None else ""
                else:
                    tgt_id = child.get("id") or ""
                    src_el = desc.find(f"{{{BPMN_NS}}}sourceRef")
                    src_id = (src_el.text or "").strip() if src_el is not None else ""
                src_shape = _find_shape_in_plane(plane, src_id)
                if src_shape is None:
                    src_shape = original_shapes.get(src_id)
                tgt_shape = _find_shape_in_plane(plane, tgt_id)
                if tgt_shape is None:
                    tgt_shape = original_shapes.get(tgt_id)
                if src_shape is not None and tgt_shape is not None:
                    wp = _docked_waypoints(src_shape, tgt_shape)
                    if wp is not None:
                        plane.append(_make_edge(did, wp))
                        emitted_frag_ids.add(did)

    return diagram


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
