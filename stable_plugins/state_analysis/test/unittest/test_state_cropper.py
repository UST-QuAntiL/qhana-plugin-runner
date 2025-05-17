import numpy as np
import pytest
from common.state_cropper import generate_reduced_substatevectors

# Test data for generate_reduced_substatevectors.
# Each dictionary represents a test case and should contain:
#   - id: A unique identifier for the test case.
#   - qasmfilecontent: The QASM code as a string.
#   - metadata: A list of intervals indicating which substate vector(s) to extract.
#   - expected: A list of expected numpy arrays representing the substate(s).
#       For error cases, this should be set to None.
#   - throws (optional): If True, the test expects an exception.
#
# Assumes that for all qubit_intervals [a_i, b_i] in qubit_intervals:
#   - a_i <= b_i
#   - 0 <= a_0 and b_i < a_i+1
#
# This method only checks if a_n <= max qubit index; the rest must be validated beforehand,
# e.g., at the interface where user input is provided.
#
# Therefore, incorrect interval inputs are not tested here but in separate tests
# that check if invalid user inputs are correctly detected.
testdata = [
    # Whole state extraction
    {
        "id": 0,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "metadata": [[0, 3]],
        "expected": [
            np.array(
                [
                    # first block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # second block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # third block
                    1 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # fourth block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                ]
            )
        ],
    },
    # Extract middle part of state (not starting at 0, not ending at 3)
    {
        "id": 1,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "metadata": [[1, 2]],
        "expected": [
            np.array([1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]),
        ],
    },
    # Extract two distinct parts of the state
    {
        "id": 2,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "metadata": [[0, 1], [2, 3]],
        "expected": [
            np.array([1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]),
            np.array([0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j]),
        ],
    },
    # Extract a small part of state (single index)
    {
        "id": 3,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "metadata": [[2, 2]],
        "expected": [
            np.array([1 + 0j, 0 + 0j]),
        ],
    },
    # Extract only the element at index 3
    {
        "id": 4,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "metadata": [[3, 3]],
        "expected": [
            np.array([0 + 0j, 1 + 0j]),
        ],
    },
    # Multiple small parts extraction
    {
        "id": 5,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "metadata": [[0, 0], [1, 1], [2, 2], [3, 3]],
        "expected": [
            np.array([1 + 0j, 0 + 0j]),
            np.array([1 + 0j, 0 + 0j]),
            np.array([1 + 0j, 0 + 0j]),
            np.array([0 + 0j, 1 + 0j]),
        ],
    },
    # Extraction from an entangled state (partial)
    {
        "id": 6,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        h q[1];
        cx q[1], q[2];
        """,
        "metadata": [[1, 2]],
        "expected": [(1 / np.sqrt(2)) * np.array([1, 0, 0, 1])],
    },
    # Extraction from a larger entangled state
    {
        "id": 7,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        h q[1];
        cx q[1], q[2];
        """,
        "metadata": [[0, 2]],
        "expected": [(1 / np.sqrt(2)) * np.array([1, 0, 0, 0, 0, 0, 1, 0])],
    },
    # Error case: entanglement crossing extraction boundaries should raise an exception.
    {
        "id": 8,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        h q[1];
        cx q[1], q[2];
        """,
        "metadata": [[0, 1]],
        "expected": None,
        "throws": True,
    },
    # Basic case: 2-qubit register, no gates applied (state |00>)
    {
        "id": 9,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        """,
        "metadata": [[0, 0], [1, 1]],
        "expected": [
            np.array([1 + 0j, 0 + 0j]),
            np.array([1 + 0j, 0 + 0j]),
        ],
    },
    # Basic case: 2-qubit register with one qubit flipped
    {
        "id": 10,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[0];
        """,
        "metadata": [[0, 1]],
        "expected": [np.array([0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j])],
    },
    # Basic case: 2-qubit register with both qubits flipped
    {
        "id": 11,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[0];
        x q[1];
        """,
        "metadata": [[0, 1]],
        "expected": [np.array([0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j])],
    },
    # Three-qubit register with no gates applied (state |000>)
    {
        "id": 12,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        // No gates applied -> state |000>
        """,
        "metadata": [[0, 2]],
        "expected": [
            np.array([1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])
        ],
    },
    # Error case: Interva exceeds the last qubit index
    {
        "id": 13,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        h q[1];
        cx q[1], q[2];
        """,
        "metadata": [[0, 100]],
        "expected": None,
        "throws": True,
    },
    # Error case: Qubit indices out of range because its smaller than 0
    {
        "id": 14,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        h q[1];
        cx q[1], q[2];
        """,
        "metadata": [[50, 3]],
        "expected": None,
        "throws": True,
    },
    # Explanation for Test Case 15:
    #
    # The QASM code defines a 6-qubit quantum circuit where:
    # - A Hadamard gate is applied to qubit q[0], creating a superposition state:
    #   (1/sqrt(2)) (|0> + |1>)
    # - X gates are applied to qubits q[2] and q[4], flipping their states from |0> to |1>.
    #
    # The qubit order in OpenQASM follows the convention where q[5] is the most significant qubit
    # and q[0] is the least significant. The overall state of the system is represented as:
    # ( |0> tensor |0>) tensor ( |0> tensor |1>) tensor ( (1/sqrt(2)) (|0> + |1>) tensor |0> )
    #
    # The metadata defines intervals focusing on specific qubit pairs:
    # - Interval [0,1] corresponds to ( (1/sqrt(2)) (|0> + |1>) tensor |0> ), resulting in:
    #   (1/sqrt(2)) [1, 1, 0, 0]
    # - Interval [2,3] corresponds to ( |0> tensor |1> ), which gives [0, 1, 0, 0].
    # - Interval [4,5] corresponds to ( |0> tensor |0> ), which also gives [0, 1, 0, 0].
    #
    # These results match the expected substate vectors.
    {
        "id": 15,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        h q[0];
        x q[2];
        x q[4];
        """,
        "metadata": [[0, 1], [2, 3], [4, 5]],
        "expected": [
            ((1 / np.sqrt(2))) * np.array([1, 1, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0, 1, 0, 0]),
        ],
    },
    # Whole state extraction metadata empty
    {
        "id": 16,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "metadata": [],
        "expected": [
            np.array(
                [
                    # first block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # second block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # third block
                    1 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # fourth block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                ]
            )
        ],
    },  # Whole state extraction metadata None
    {
        "id": 17,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "metadata": None,
        "expected": [
            np.array(
                [
                    # first block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # second block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # third block
                    1 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    # fourth block
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                    0 + 0j,
                ]
            )
        ],
    },
]


@pytest.mark.parametrize("case", testdata, ids=[f"case_{d['id']}" for d in testdata])
def test_generate_reduced_substatevectors(case):
    """
    Parameterized test for generate_reduced_substatevectors using various QASM inputs and metadata.
    """
    qasm_code = case["qasmfilecontent"]
    metadata = case["metadata"]
    expected = case.get("expected")
    should_throw = case.get("throws", False)
    case_id = case.get("id")

    if should_throw:
        # Verify that an exception is raised for error cases.
        with pytest.raises(Exception):
            generate_reduced_substatevectors(qasm_code, metadata)
    else:
        # Execute the function and compare output with expected results.
        output = generate_reduced_substatevectors(qasm_code, metadata)

        # Check that the number of returned substate vectors matches the expectation.
        assert len(output) == len(
            expected
        ), f"Test case {case_id}: Expected {len(expected)} substate vector(s), got {len(output)}"

        # Compare each output vector to its corresponding expected array.
        for idx, (out_vec, exp_vec) in enumerate(zip(output, expected)):
            assert np.allclose(
                out_vec, exp_vec, atol=1e-6
            ), f"Test case {case_id}: Mismatch in substate vector {idx}: {out_vec} != {exp_vec}"
