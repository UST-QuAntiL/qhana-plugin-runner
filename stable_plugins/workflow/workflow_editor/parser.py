from collections import deque
from dataclasses import dataclass, field
from json import dumps
from textwrap import indent
from types import SimpleNamespace
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, TypeAlias, Union
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

BPMN_NS = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"
BPMN_DI_NS = "{http://www.omg.org/spec/BPMN/20100524/DI}"

CAMUNDA_NS = "{http://camunda.org/schema/1.0/bpmn}"

QUANT_ME_NS = "{https://github.com/UST-QuAntiL/QuantME-Quantum4BPMN}"

PLANQK_NS = "{https://platform.planqk.de}"

QHANA_NS = "{https://github.com/qhana}"

TRANSFORMATION_NS = "{https://github.com/data/transformation}"

IGNORED_TAGS = {
    f"{BPMN_DI_NS}BPMNDiagram",
    f"{BPMN_NS}incoming",
    f"{BPMN_NS}outgoing",
    f"{BPMN_NS}documentation",
    f"{BPMN_NS}definitions",
    f"{BPMN_NS}textAnnotation",
    f"{BPMN_NS}text",
    f"{BPMN_NS}extensionElements",
    f"{BPMN_NS}dataObject",
    f"{BPMN_NS}dataObjectReference",
    f"{BPMN_NS}dataOutputAssociation",
    f"{BPMN_NS}targetRef",
    f"{BPMN_NS}map",
    f"{BPMN_NS}inputParameter",
    f"{BPMN_NS}entry",
    f"{CAMUNDA_NS}inputOutput",
    f"{CAMUNDA_NS}properties",
    f"{TRANSFORMATION_NS}dataMapObject",
    f"{TRANSFORMATION_NS}keyValueEntry",
}

BPMN = SimpleNamespace(
    definitions=f"{BPMN_NS}definitions",
    process=f"{BPMN_NS}process",
    text=f"{BPMN_NS}text",
    textAnnotation=f"{BPMN_NS}textAnnotation",
    association=f"{BPMN_NS}association",
    sequenceFlow=f"{BPMN_NS}sequenceFlow",
    startEvent=f"{BPMN_NS}startEvent",
    endEvent=f"{BPMN_NS}endEvent",
    subProcess=f"{BPMN_NS}subProcess",
    adHocSubProcess=f"{BPMN_NS}adHocSubProcess",
    task=f"{BPMN_NS}task",
    serviceTask=f"{BPMN_NS}serviceTask",
    scriptTask=f"{BPMN_NS}scriptTask",
    manualTask=f"{BPMN_NS}manualTask",
    userTask=f"{BPMN_NS}userTask",
    exclusiveGateway=f"{BPMN_NS}exclusiveGateway",
    parallelGateway=f"{BPMN_NS}parallelGateway",
    # TODO fill out
)

QUANT_ME = SimpleNamespace(
    quantumComputationTask=f"{QUANT_ME_NS}quantumComputationTask",
)

PLANQK = SimpleNamespace(
    serviceTask=f"{PLANQK_NS}serviceTask",
)

QHANA = SimpleNamespace(
    serviceTask=f"{QHANA_NS}qHAnaServiceTask",
)

START_EVENTS = {
    BPMN.startEvent,
}


@dataclass()
class ActivityLike:
    xml: Element
    id_: Optional[str]
    type_: str
    parent: Optional["ActivityLike"] = None
    start_event: Optional["EventLike"] = None
    children: Sequence["ActivityLike"] = tuple()
    incoming: Sequence["FlowLike"] = tuple()
    outgoing: Sequence["FlowLike"] = tuple()


@dataclass()
class GateLike:
    xml: Element
    id_: Optional[str]
    type_: str
    incoming: Sequence["FlowLike"] = tuple()
    outgoing: Sequence["FlowLike"] = tuple()


@dataclass()
class EventLike:
    xml: Element
    id_: Optional[str]
    type_: str
    incoming: Sequence["FlowLike"] = tuple()
    outgoing: Sequence["FlowLike"] = tuple()


@dataclass()
class FlowLike:
    xml: Element
    id_: Optional[str]
    type_: str
    source: Optional[Union[ActivityLike, GateLike, EventLike]] = field(
        default=None, repr=False
    )
    target: Optional[Union[ActivityLike, GateLike, EventLike]] = field(
        default=None, repr=False
    )


@dataclass()
class NodeGroup:
    id_: int
    nodes: set[str]
    kind: Literal["unknown", "neutral", "executable", "ad-hoc", "mixed"] = "unknown"
    incoming: Sequence["FlowLike"] = tuple()
    outgoing: Sequence["FlowLike"] = tuple()


@dataclass()
class ParsedWorkflow:
    root: Element
    start_event: Optional[EventLike] = None
    flows: dict[str, FlowLike] = field(default_factory=dict)
    activities: dict[str, ActivityLike] = field(default_factory=dict)
    gates: dict[str, GateLike] = field(default_factory=dict)
    events: dict[str, EventLike] = field(default_factory=dict)
    element_by_id: dict[str, FlowLike | ActivityLike | GateLike | EventLike] = field(
        default_factory=dict
    )


@dataclass()
class UiTemplateTaskGroup:
    element: ActivityLike | GateLike
    outgoing: List["UiTemplateTaskGroup"] = field(default_factory=list)
    children: List["UiTemplateTaskGroup"] = field(default_factory=list)
    plugin_filter: Optional[dict] = None

    @property
    def id_(self) -> str:
        return self.element.id_

    @property
    def name(self) -> str:
        if isinstance(self.element, GateLike):
            if self.element.type_ == BPMN.exclusiveGateway:
                return "XOR"
            elif self.element.type_ == BPMN.parallelGateway:
                return "AND"
            return "Gate " + self.element.type_.removeprefix(BPMN_NS)
        return self.element.xml.attrib.get("name", self.element.id_)

    @property
    def description(self) -> str:
        return ""  # FIXME extract from element

    def __str__(self) -> str:
        outgoing = ""
        if self.outgoing:
            outgoing = f", → {len(self.outgoing)}: "
            outgoing += " ".join(g.id_ for g in self.outgoing)
        text = f"{self.name} ({self.id_}{outgoing})"
        if self.plugin_filter:
            text += f"\n> {self.plugin_filter}"
        if self.children:
            text += "\n"
            text += indent("\n".join(str(c) for c in self.children), "    ")
        return text


def split_ui_template_workflow(bpmn: str) -> tuple[str, tuple[str, ...]]:
    # root = ElementTree.fromstring(bpmn)

    # TODO split workflow into a main workflow containing only ad-hoc task groups for the UI template
    # and 0 or more executable workflows that need to be deployed as plugins.
    # The main workflow should have an ad-hoc group with the plugin as placeholder for the extracted executable parts.

    return bpmn, tuple()


def get_ad_hoc_tree(bpmn: str) -> Sequence[UiTemplateTaskGroup]:
    parsed = _parse_bpmn(bpmn)
    return _extract_groups(parsed, parsed.start_event)


def _extract_groups(  # noqa: C901
    workflow: ParsedWorkflow, start: EventLike, visited: Optional[Set[str]] = None
) -> Sequence[UiTemplateTaskGroup]:
    if visited is None:
        visited = set()

    groups_by_id: Dict[str, UiTemplateTaskGroup] = {}
    groups_flat: List[UiTemplateTaskGroup] = []

    element_queue: TypeAlias = deque[
        Tuple[EventLike | ActivityLike | GateLike, Optional[UiTemplateTaskGroup]]
    ]
    # the stack is used when elements on the same level in the BPMN graph
    # should be rendered as children to the current group (i.e., elements betweenan OR gate)
    queue_stack: list[
        Tuple[UiTemplateTaskGroup, List[UiTemplateTaskGroup], element_queue]
    ] = []
    queue: element_queue = deque([(start, None)])
    while queue:
        element, predecessor = queue.popleft()
        visited.add(element.id_)

        current_group = predecessor

        if isinstance(element, GateLike):
            # handle gates like left/right brackets (using the stack)
            assert (
                len(element.incoming) == 1 or len(element.outgoing) == 1
            ), "complex gates are not supported"
            found_match = False
            if queue_stack and len(element.incoming) >= 1 and len(element.outgoing) == 1:
                # right bracket gate
                parent_group = queue_stack[-1][0]
                if parent_group.element.type_ == element.type_:
                    if queue:
                        # there are still elements in the queue,
                        # cannot mark right bracket as processed until
                        # all elements in the queue are processed
                        visited.discard(element.id_)
                        continue
                    else:
                        # go one level up
                        current_group, groups_flat, queue = queue_stack.pop()
                        found_match = True
            if (
                len(element.outgoing) >= 1
                and len(element.incoming) == 1
                and not found_match
            ):
                # left bracket gate
                assert (
                    element.id_ not in groups_by_id
                ), "left bracket gates canonly have one input flow"
                current_group = UiTemplateTaskGroup(element)
                groups_by_id[element.id_] = current_group
                groups_flat.append(current_group)
                if predecessor:
                    predecessor.outgoing.append(current_group)

                # go one layer deeper
                queue_stack.append((current_group, groups_flat, queue))
                groups_flat = current_group.children
                queue = deque()
                current_group = None

        # the following is focused on handling standard activities
        elif isinstance(element, ActivityLike) and element.type_ in (
            BPMN.adHocSubProcess,
            BPMN.subProcess,
        ):
            if element.id_ in groups_by_id:
                current_group = groups_by_id[element.id_]
            else:
                current_group = UiTemplateTaskGroup(element)
                _fill_plugin_filter(current_group)
                groups_by_id[element.id_] = current_group
                groups_flat.append(current_group)
            if predecessor:
                predecessor.outgoing.append(current_group)
        for flow in element.outgoing:
            if flow.target is not None and flow.target.id_ not in visited:
                if not isinstance(flow.target, EventLike):
                    queue.appendleft((flow.target, current_group))
        if isinstance(element, (EventLike, GateLike)):
            continue
        elif element.type_ == BPMN.subProcess:
            assert (
                element.start_event is not None
            ), "a subprocess must have an associated start event"
            current_group.children = _extract_groups(
                workflow, element.start_event, visited
            )

    return groups_flat


def _fill_plugin_filter(group: UiTemplateTaskGroup):
    if not isinstance(group.element, ActivityLike):
        return
    if group.element.type_ != BPMN.adHocSubProcess:
        return
    children = group.element.children

    plugin_filter = {"or": []}

    for child in children:
        assert isinstance(child, ActivityLike)
        assert child.type_ in (QHANA.serviceTask, BPMN.serviceTask)

        attributes = {}
        attributes.update(child.xml.attrib)

        for i in child.xml.findall(
            ".//{http://camunda.org/schema/1.0/bpmn}inputParameter"
        ):
            if i.attrib.get("name", "").startswith("qhana"):
                attributes[i.attrib["name"]] = i.text.strip() if i.text else ""
        match attributes:
            case {"qhanaIdentifier": identifier, "qhanaVersion": version} if (
                identifier.strip() and version.strip()
            ):
                plugin_filter["or"].append({"id": f"{identifier}@{version}"})
            case {"qhanaIdentifier": identifier} if identifier.strip():
                plugin_filter["or"].append({"id": identifier})
            case {"qhanaName": name} if name.strip():
                plugin_filter["or"].append({"name": name})
            case _:
                print(child.xml.attrib, attributes)

    group.plugin_filter = plugin_filter


def tree_to_template_tabs(
    groups: Sequence[UiTemplateTaskGroup],
    path: Sequence[str] = tuple(),
    base_location: str = "experiment-navigation",
):
    tabs = []

    location = base_location
    if path:
        location += "." + ".".join(path)

    for i, group in enumerate(groups):
        suffix = " →" if group.outgoing else ""

        group_key = ""
        if group.children:
            group_key = f"{i+1}"

        plugin_filter = ""
        if group.plugin_filter and not group.children:
            plugin_filter = dumps(group.plugin_filter)

        tab = {
            "name": f"{i+1}. {group.name}{suffix}",
            "description": group.description,
            "icon": "",  # TODO, better icons? specified in WF?
            "location": location,
            "sortKey": i,
            "groupKey": group_key,
            "filterString": plugin_filter,
        }
        tabs.append(tab)

        if group.children:
            tabs.extend(
                tree_to_template_tabs(
                    group.children, path=(*path, group_key), base_location=base_location
                )
            )
    return tabs


def _parse_bpmn(bpmn: str):
    parsed = ParsedWorkflow(ElementTree.fromstring(bpmn))

    start_events: Dict[str, EventLike] = {}

    nodes = deque([(parsed.root, parsed.root)])
    while len(nodes) > 0:
        node, parent = nodes.popleft()
        next_parent = parent
        if node.tag == BPMN.subProcess:
            next_parent = node
        for n in node:
            if n.tag in IGNORED_TAGS:
                continue
            nodes.append((n, next_parent))
        match node:
            case Element(tag=BPMN.process):
                pass
            case Element(tag=BPMN.definitions):
                pass
            case Element(tag=BPMN.association):
                pass
            case Element(tag=BPMN.sequenceFlow, attrib={"id": id_}):
                parsed.element_by_id[id_] = parsed.flows[id_] = FlowLike(
                    node, id_, node.tag
                )
            case Element(tag=BPMN.startEvent, attrib={"id": id_}):
                event = EventLike(node, id_, node.tag)
                if parent == parsed.root:
                    assert (
                        parsed.start_event is None
                    ), "a workflow can only have a single start event"
                    parsed.start_event = event
                else:
                    start_events[parent.attrib["id"]] = event
                parsed.element_by_id[id_] = parsed.events[id_] = event
            case Element(tag=BPMN.endEvent, attrib={"id": id_}):
                parsed.element_by_id[id_] = parsed.events[id_] = EventLike(
                    node, id_, node.tag
                )
            case Element(
                tag=(
                    BPMN.adHocSubProcess
                    | BPMN.subProcess
                    | BPMN.task
                    | BPMN.serviceTask
                    | BPMN.scriptTask
                    | BPMN.manualTask
                    | BPMN.userTask
                    | PLANQK.serviceTask
                    | QUANT_ME.quantumComputationTask
                    | QHANA.serviceTask
                ),
                attrib={"id": id_},
            ):
                parsed.element_by_id[id_] = parsed.activities[id_] = ActivityLike(
                    node, id_, node.tag
                )
            case Element(
                tag=(BPMN.exclusiveGateway | BPMN.parallelGateway), attrib={"id": id_}
            ):
                parsed.element_by_id[id_] = parsed.gates[id_] = GateLike(
                    node, id_, node.tag
                )
            case _:
                print(node.tag, node.attrib)

    # postprocess flows
    for flow in parsed.flows.values():
        match flow.xml.attrib:
            case {"sourceRef": source_id, "targetRef": target_id}:
                source = parsed.element_by_id.get(source_id)
                target = parsed.element_by_id.get(target_id)
                assert isinstance(
                    source, (ActivityLike, EventLike, GateLike)
                ), "source must not be none and must not be a flow"
                assert isinstance(
                    target, (ActivityLike, EventLike, GateLike)
                ), "target must not be none and must not be a flow"
                flow.source = source
                flow.target = target
                source.outgoing = (*source.outgoing, flow)
                target.incoming = (*target.incoming, flow)
            case _:
                print(flow.xml.attrib)

    # postrpocess start events
    for parent_id, start_event in start_events.items():
        parent = parsed.activities[parent_id]
        assert parent.type_ == BPMN.subProcess
        parent.start_event = start_event

    # postprocess activities
    for activity in parsed.activities.values():
        for node in activity.xml:
            match node:
                case Element(attrib={"id": id_}) if id_ in parsed.activities:
                    child = parsed.activities[id_]
                    assert child.parent is None
                    child.parent = activity
                    activity.children = (*activity.children, child)

    return parsed


if __name__ == "__main__":  # TODO remove later
    from pathlib import Path

    bpmn = Path(
        # "stable_plugins/workflow/workflow_editor/assets/ui-template-demo.bpmn"
        "stable_plugins/workflow/workflow_editor/assets/ui-template-demo-transformed.bpmn"
    ).read_text()

    groups = get_ad_hoc_tree(bpmn)

    for g in groups:
        print(g)

    for t in tree_to_template_tabs(groups):
        print(t)
