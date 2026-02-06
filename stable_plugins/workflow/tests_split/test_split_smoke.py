from pathlib import Path
import pytest
import xml.etree.ElementTree as ET

from splitting.split import split_workflow, SplitNotSupported
ROOT = Path(__file__).resolve().parents[0]  # tests_split/
WF_EDITOR = ROOT.parent / "workflow_editor"
BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
NS = {"bpmn": BPMN_NS}

def test_split_smoke_tc01_roundtrip():
    bpmn = (WF_EDITOR / "tests" / "bpmn" / "tc01_hello_world_adhoc.bpmn").read_text(encoding="utf-8")
    exec_xml, rest_xml = split_workflow(bpmn)
    
    assert "_exec" in exec_xml
    assert "_rest" in rest_xml

    # exec should contain the hello-world task (either qhana tag or transformed serviceTask)
    assert ("qHAnaServiceTask" in exec_xml) or ("qhanaPlugin" in exec_xml)

    # rest should still contain the manual input step
    assert "UserTask_Input" in rest_xml or "Input (UserTask)" in rest_xml
    # print("\n--- EXEC BPMN ---\n", exec_xml)
    # print("\n--- REST BPMN ---\n", rest_xml)
    Path("tc01_exec.bpmn").write_text(exec_xml, encoding="utf-8")
    Path("tc01_rest.bpmn").write_text(rest_xml, encoding="utf-8")


