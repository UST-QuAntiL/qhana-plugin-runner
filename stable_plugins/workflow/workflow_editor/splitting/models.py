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
