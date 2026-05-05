import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Optional

from .models import (
    SplitNotSupported,
    SplitResult,
    FragmentResult,
    default_classifier,
)
from .classify import (
    _find_process,
    _parse_process,
    _classify_nodes,
    _find_regions,
)
from .build import (
    _build_main_process,
    _build_fragment_process,
    _compute_region_io,
    _ensure_history_ttl,
)
from .di import (
    _build_main_definitions,
    _build_fragment_definitions,
    _serialize,
)


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

    _ensure_history_ttl(main_process)

    main_defs = _build_main_definitions(
        original_root, main_process, original_pid, new_main_pid, regions
    )
    main_xml = _serialize(main_defs)

    fragments: List[FragmentResult] = []
    for r in regions:
        fid = f"E{r.index}"
        frag_pid = f"{original_pid}_{fid}"
        frag_proc = _build_fragment_process(r, nodes, flows, frag_pid, original_process)
        _ensure_history_ttl(frag_proc)
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
