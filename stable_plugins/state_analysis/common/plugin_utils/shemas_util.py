from common.plugin_utils.marshmallow_util import QasmInputList

# This is used in every classical plugin so to make changes more easy this is used as reference

qasmInputList_util = QasmInputList(
    required=False,
    allow_none=True,
    # TODO ui is not working if this is commented in
    # data_input_type="executable/circuit",
    # data_content_types="text/x-qasm",
    metadata={
        "label": "OpenQASM Circuit",
        "description": (
            "A list of lists: "
            " The first element is a URL to a QASM file (required). "
            " The second element is an json with qubit_intervals (may be None)."
            ' Example: [["https://example.com/qasm1.qasm", {"qubit_intervals": [[1, 5],[7, 10],[14, 15]]}],["https://example.com/qasm2.qasm"]]'
        ),
        "input_type": "textarea",
    },
)
