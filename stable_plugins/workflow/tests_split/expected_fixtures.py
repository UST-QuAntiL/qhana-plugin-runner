EXPECTED = {
    "tc01_exec_before_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_1"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_4", "AdHoc_E1_Wrapper", "AdHocSubProcess_1", None),
            ("Flow_5", "AdHocSubProcess_1", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc01_exec_before_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.analysisResult"],
                "task_ids": ["UserTask_Input", "QHanaTask_Prepare", "QHanaTask_Analyze"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3", "Flow_E1_end"],
            }
        ],
    },
    "tc02_exec_after_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[original]", "AdHocSubProcess_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHocSubProcess_1", None),
            ("Flow_2", "AdHocSubProcess_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc02_exec_after_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": ["qoutput.filteredText"],
                "outputs": [],
                "task_ids": ["QHanaTask_Process", "QHanaTask_Summarize"],
                "flow_ids": ["Flow_E1_start", "Flow_3", "Flow_4"],
            }
        ],
    },
    "tc03_exec_before_and_after_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_1"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_4", "AdHoc_E1_Wrapper", "AdHocSubProcess_1", None),
            ("Flow_5", "AdHocSubProcess_1", "AdHoc_E2_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E2_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc03_exec_before_and_after_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.normalizedDataset"],
                "task_ids": ["UserTask_Input", "QHanaTask_Load", "QHanaTask_Normalize"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc03_exec_before_and_after_adhoc_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.annotatedDataset"],
                "outputs": [],
                "task_ids": ["QHanaTask_Transform", "QHanaTask_Save"],
                "flow_ids": ["Flow_E2_start", "Flow_6", "Flow_7"],
            },
        ],
    },
    "tc04_only_exec_no_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc04_only_exec_no_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "UserTask_Input",
                    "QHanaTask_Fetch",
                    "QHanaTask_Classify",
                    "QHanaTask_Store",
                    "QHanaTask_Notify",
                ],
                "flow_ids": [
                    "Flow_E1_start",
                    "Flow_2",
                    "Flow_3",
                    "Flow_4",
                    "Flow_5",
                    "Flow_6",
                ],
            }
        ],
    },
    "tc05_only_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[original]", "AdHocSubProcess_1"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHocSubProcess_1", None),
            ("Flow_2", "AdHocSubProcess_1", "EndEvent_1", None),
        ],
        "fragments": [],
    },
    "tc06_multiple_adhoc_blocks.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_1"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_2"),
            ("adHocSubProcess[wrapper=E3]", "AdHoc_E3_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "AdHocSubProcess_1", None),
            ("Flow_3", "AdHocSubProcess_1", "AdHoc_E2_Wrapper", None),
            ("Flow_4", "AdHoc_E2_Wrapper", "AdHocSubProcess_2", None),
            ("Flow_5", "AdHocSubProcess_2", "AdHoc_E3_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E3_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc06_multiple_adhoc_blocks_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.loadedData"],
                "task_ids": ["QHanaTask_Load"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc06_multiple_adhoc_blocks_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.annotatedData"],
                "outputs": ["qoutput.transformedData"],
                "task_ids": ["QHanaTask_Transform"],
                "flow_ids": ["Flow_E2_start", "Flow_E2_end"],
            },
            {
                "fragment_id": "E3",
                "process_id": "tc06_multiple_adhoc_blocks_E3",
                "wrapper_id": "AdHoc_E3_Wrapper",
                "inputs": ["qoutput.approvedValidation"],
                "outputs": [],
                "task_ids": ["QHanaTask_Save"],
                "flow_ids": ["Flow_E3_start", "Flow_6"],
            },
        ],
    },
    "tc07_xor_exec_vs_exec.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc07_xor_exec_vs_exec_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "UserTask_Input",
                    "Gateway_Split",
                    "QHanaTask_TranslateFR",
                    "QHanaTask_TranslateES",
                    "Gateway_Merge",
                ],
                "flow_ids": [
                    "Flow_E1_start",
                    "Flow_2",
                    "Flow_3A",
                    "Flow_3B",
                    "Flow_4A",
                    "Flow_4B",
                    "Flow_5",
                ],
            }
        ],
    },
    "tc08_xor_exec_vs_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("exclusiveGateway", "Gateway_Split"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Manual"),
            ("exclusiveGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "Gateway_Split", None),
            ("Flow_3A", "Gateway_Split", "AdHoc_E2_Wrapper", "${amount < 10000}"),
            ("Flow_3B", "Gateway_Split", "AdHocSubProcess_Manual", None),
            ("Flow_4A", "AdHoc_E2_Wrapper", "Gateway_Merge", None),
            ("Flow_4B", "AdHocSubProcess_Manual", "Gateway_Merge", None),
            ("Flow_5", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc08_xor_exec_vs_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.applicationData"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc08_xor_exec_vs_adhoc_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.applicationData"],
                "outputs": [],
                "task_ids": ["QHanaTask_AutoScore"],
                "flow_ids": ["Flow_E2_start", "Flow_E2_end"],
            },
        ],
    },
    "tc09_xor_adhoc_vs_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("exclusiveGateway", "Gateway_Split"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Tier1"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Tier2"),
            ("exclusiveGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "Gateway_Split", None),
            ("Flow_3A", "Gateway_Split", "AdHocSubProcess_Tier1", "${tier == 'Tier1'}"),
            ("Flow_3B", "Gateway_Split", "AdHocSubProcess_Tier2", None),
            ("Flow_4A", "AdHocSubProcess_Tier1", "Gateway_Merge", None),
            ("Flow_4B", "AdHocSubProcess_Tier2", "Gateway_Merge", None),
            ("Flow_5", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc09_xor_adhoc_vs_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.ticketData"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            }
        ],
    },
    "tc10_xor_exec_before_merge_then_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("exclusiveGateway", "Gateway_Merge"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Review"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_5B", "AdHoc_E1_Wrapper", "Gateway_Merge", None),
            ("Flow_6", "Gateway_Merge", "AdHocSubProcess_Review", None),
            ("Flow_7", "AdHocSubProcess_Review", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc10_xor_exec_before_merge_then_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.algoResult"],
                "task_ids": [
                    "UserTask_Input",
                    "QHanaTask_CleanData",
                    "Gateway_Split",
                    "QHanaTask_AlgoA",
                    "QHanaTask_AlgoB",
                ],
                "flow_ids": [
                    "Flow_E1_start",
                    "Flow_2",
                    "Flow_3",
                    "Flow_4A",
                    "Flow_4B",
                    "Flow_E1_end_0",
                    "Flow_E1_end_1",
                ],
            }
        ],
    },
    "tc11_xor_adhoc_inside_branch.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("exclusiveGateway", "Gateway_Split"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Manual"),
            ("adHocSubProcess[wrapper=E3]", "AdHoc_E3_Wrapper"),
            ("adHocSubProcess[wrapper=E4]", "AdHoc_E4_Wrapper"),
            ("exclusiveGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "Gateway_Split", None),
            ("Flow_3A", "Gateway_Split", "AdHoc_E2_Wrapper", "${riskLevel == 'High'}"),
            ("Flow_4A", "AdHoc_E2_Wrapper", "AdHocSubProcess_Manual", None),
            ("Flow_5A", "AdHocSubProcess_Manual", "AdHoc_E3_Wrapper", None),
            ("Flow_6A", "AdHoc_E3_Wrapper", "Gateway_Merge", None),
            ("Flow_3B", "Gateway_Split", "AdHoc_E4_Wrapper", None),
            ("Flow_6B", "AdHoc_E4_Wrapper", "Gateway_Merge", None),
            ("Flow_7", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc11_xor_adhoc_inside_branch_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.contentUrl"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc11_xor_adhoc_inside_branch_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.contentUrl"],
                "outputs": ["qoutput.contentMeta"],
                "task_ids": ["QHanaTask_ExtractMeta"],
                "flow_ids": ["Flow_E2_start", "Flow_E2_end"],
            },
            {
                "fragment_id": "E3",
                "process_id": "tc11_xor_adhoc_inside_branch_E3",
                "wrapper_id": "AdHoc_E3_Wrapper",
                "inputs": ["qoutput.contentMeta"],
                "outputs": [],
                "task_ids": ["QHanaTask_ApplyWarnings"],
                "flow_ids": ["Flow_E3_start", "Flow_E3_end"],
            },
            {
                "fragment_id": "E4",
                "process_id": "tc11_xor_adhoc_inside_branch_E4",
                "wrapper_id": "AdHoc_E4_Wrapper",
                "inputs": ["qoutput.contentUrl"],
                "outputs": [],
                "task_ids": ["QHanaTask_AutoModerate"],
                "flow_ids": ["Flow_E4_start", "Flow_E4_end"],
            },
        ],
    },
    "tc12_xor_default_flow_preservation.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("exclusiveGateway", "Gateway_Split"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Backorder"),
            ("exclusiveGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "Gateway_Split", None),
            ("Flow_3A", "Gateway_Split", "AdHoc_E2_Wrapper", None),
            (
                "Flow_3B",
                "Gateway_Split",
                "AdHocSubProcess_Backorder",
                "${stock == 'Empty'}",
            ),
            ("Flow_4A", "AdHoc_E2_Wrapper", "Gateway_Merge", None),
            ("Flow_4B", "AdHocSubProcess_Backorder", "Gateway_Merge", None),
            ("Flow_5", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc12_xor_default_flow_preservation_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.orderData"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc12_xor_default_flow_preservation_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.orderData"],
                "outputs": [],
                "task_ids": ["QHanaTask_ProcessPayment"],
                "flow_ids": ["Flow_E2_start", "Flow_E2_end"],
            },
        ],
    },
    "tc13_and_exec_parallel.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc13_and_exec_parallel_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "UserTask_Input",
                    "Gateway_Split",
                    "QHanaTask_ITProvision",
                    "QHanaTask_PayrollSetup",
                    "Gateway_Merge",
                ],
                "flow_ids": [
                    "Flow_E1_start",
                    "Flow_2",
                    "Flow_3A",
                    "Flow_3B",
                    "Flow_4A",
                    "Flow_4B",
                    "Flow_5",
                ],
            }
        ],
    },
    "tc14_and_exec_vs_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("parallelGateway", "Gateway_Split"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Security"),
            ("parallelGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "Gateway_Split", None),
            ("Flow_3A", "Gateway_Split", "AdHoc_E2_Wrapper", None),
            ("Flow_3B", "Gateway_Split", "AdHocSubProcess_Security", None),
            ("Flow_4A", "AdHoc_E2_Wrapper", "Gateway_Merge", None),
            ("Flow_4B", "AdHocSubProcess_Security", "Gateway_Merge", None),
            ("Flow_5", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc14_and_exec_vs_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.releaseData"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc14_and_exec_vs_adhoc_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.releaseData"],
                "outputs": [],
                "task_ids": ["QHanaTask_AutoTest"],
                "flow_ids": ["Flow_E2_start", "Flow_E2_end"],
            },
        ],
    },
    "tc15_and_both_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("parallelGateway", "Gateway_Split"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Venue"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Marketing"),
            ("parallelGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "Gateway_Split", None),
            ("Flow_3A", "Gateway_Split", "AdHocSubProcess_Venue", None),
            ("Flow_3B", "Gateway_Split", "AdHocSubProcess_Marketing", None),
            ("Flow_4A", "AdHocSubProcess_Venue", "Gateway_Merge", None),
            ("Flow_4B", "AdHocSubProcess_Marketing", "Gateway_Merge", None),
            ("Flow_5", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc15_and_both_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.eventDetails"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            }
        ],
    },
    "tc16_and_join_after_extraction.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("parallelGateway", "Gateway_Split"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_ManualReview"),
            ("adHocSubProcess[wrapper=E3]", "AdHoc_E3_Wrapper"),
            ("parallelGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "Gateway_Split", None),
            ("Flow_3A", "Gateway_Split", "AdHoc_E2_Wrapper", None),
            ("Flow_4A", "AdHoc_E2_Wrapper", "Gateway_Merge", None),
            ("Flow_3B", "Gateway_Split", "AdHocSubProcess_ManualReview", None),
            ("Flow_4B", "AdHocSubProcess_ManualReview", "AdHoc_E3_Wrapper", None),
            ("Flow_5B", "AdHoc_E3_Wrapper", "Gateway_Merge", None),
            ("Flow_6", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc16_and_join_after_extraction_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.architectureData"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc16_and_join_after_extraction_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.architectureData"],
                "outputs": [],
                "task_ids": ["QHanaTask_Linter"],
                "flow_ids": ["Flow_E2_start", "Flow_E2_end"],
            },
            {
                "fragment_id": "E3",
                "process_id": "tc16_and_join_after_extraction_E3",
                "wrapper_id": "AdHoc_E3_Wrapper",
                "inputs": ["qoutput.diagramReview"],
                "outputs": [],
                "task_ids": ["QHanaTask_GenerateReport"],
                "flow_ids": ["Flow_E3_start", "Flow_E3_end"],
            },
        ],
    },
    "tc17_and_duplicate_edge_case.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("parallelGateway", "Gateway_Split"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("adHocSubProcess[wrapper=E3]", "AdHoc_E3_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Audit"),
            ("parallelGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "Gateway_Split", None),
            ("Flow_3A", "Gateway_Split", "AdHoc_E2_Wrapper", None),
            ("Flow_3B", "Gateway_Split", "AdHoc_E3_Wrapper", None),
            ("Flow_3C", "Gateway_Split", "AdHocSubProcess_Audit", None),
            ("Flow_4A", "AdHoc_E2_Wrapper", "Gateway_Merge", None),
            ("Flow_4B", "AdHoc_E3_Wrapper", "Gateway_Merge", None),
            ("Flow_4C", "AdHocSubProcess_Audit", "Gateway_Merge", None),
            ("Flow_5", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc17_and_duplicate_edge_case_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.processData"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc17_and_duplicate_edge_case_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.processData"],
                "outputs": [],
                "task_ids": ["QHanaTask_Worker1"],
                "flow_ids": ["Flow_E2_start", "Flow_E2_end"],
            },
            {
                "fragment_id": "E3",
                "process_id": "tc17_and_duplicate_edge_case_E3",
                "wrapper_id": "AdHoc_E3_Wrapper",
                "inputs": ["qoutput.processData"],
                "outputs": [],
                "task_ids": ["QHanaTask_Worker2"],
                "flow_ids": ["Flow_E3_start", "Flow_E3_end"],
            },
        ],
    },
    "tc18_adhoc_at_start.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Init"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHocSubProcess_Init", None),
            ("Flow_2", "AdHocSubProcess_Init", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc18_adhoc_at_start_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": ["qoutput.scopeData"],
                "outputs": [],
                "task_ids": ["QHanaTask_Provision"],
                "flow_ids": ["Flow_E1_start", "Flow_3"],
            }
        ],
    },
    "tc19_adhoc_at_end.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_FinalReview"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_3", "AdHoc_E1_Wrapper", "AdHocSubProcess_FinalReview", None),
            ("Flow_4", "AdHocSubProcess_FinalReview", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc19_adhoc_at_end_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.processedData"],
                "task_ids": ["UserTask_Input", "QHanaTask_ProcessData"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_E1_end"],
            }
        ],
    },
    "tc20_nested_adhoc.bpmn": {
        "nsup": "Nested ad-hoc subprocess not supported for UI template generation (outer "
        "id='AdHocSubProcess_Outer', inner id='AdHocSubProcess_Inner'). Use a standard "
        "subprocess as the outer container."
    },
    "tc21_adhoc_with_non_qhana_task.bpmn": {
        "nsup": "Ad-hoc subprocess 'AdHocSubProcess_Approval' contains non-QHAna task "
        "'ScriptTask_LogAudit' (type='scriptTask'). All tasks inside an "
        "ad-hoc must be QHAna-recognized for UI template generation."
    },
    "tc22_multiple_exec_segments_around_adhoc.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Review1"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Review2"),
            ("adHocSubProcess[wrapper=E3]", "AdHoc_E3_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_3", "AdHoc_E1_Wrapper", "AdHocSubProcess_Review1", None),
            ("Flow_4", "AdHocSubProcess_Review1", "AdHoc_E2_Wrapper", None),
            ("Flow_5", "AdHoc_E2_Wrapper", "AdHocSubProcess_Review2", None),
            ("Flow_6", "AdHocSubProcess_Review2", "AdHoc_E3_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E3_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc22_multiple_exec_segments_around_adhoc_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.draft"],
                "task_ids": ["UserTask_Input", "QHanaTask_GenDraft"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc22_multiple_exec_segments_around_adhoc_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.reviewedDraft"],
                "outputs": ["qoutput.plagReport"],
                "task_ids": ["QHanaTask_PlagCheck"],
                "flow_ids": ["Flow_E2_start", "Flow_E2_end"],
            },
            {
                "fragment_id": "E3",
                "process_id": "tc22_multiple_exec_segments_around_adhoc_E3",
                "wrapper_id": "AdHoc_E3_Wrapper",
                "inputs": ["qoutput.approvalLog"],
                "outputs": [],
                "task_ids": ["QHanaTask_Publish"],
                "flow_ids": ["Flow_E3_start", "Flow_7"],
            },
        ],
    },
    "tc23_mixed_task_types_exec_chain.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc23_mixed_task_types_exec_chain_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": ["scriptValue"],
                "outputs": [],
                "task_ids": [
                    "UserTask_Collect",
                    "ServiceTask_Resolve",
                    "ScriptTask_Prepare",
                    "BusinessRuleTask_Decide",
                    "QHanaTask_Finalize",
                ],
                "flow_ids": [
                    "Flow_E1_start",
                    "Flow_2",
                    "Flow_3",
                    "Flow_4",
                    "Flow_5",
                    "Flow_6",
                ],
            }
        ],
    },
    "tc24_non_convertible_tasks.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc24_non_convertible_tasks_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "QHanaTask_Prepare",
                    "ServiceTask_Unknown",
                    "QHanaTask_Finalize",
                ],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3", "Flow_4"],
            }
        ],
    },
    "tc25_single_user_task.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc25_user_task_as_exec_candidate_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "UserTask_SelectOption",
                    "QHanaTask_RunSelection",
                    "QHanaTask_SaveSelection",
                ],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3", "Flow_4"],
            }
        ],
    },
    "tc26_variable_exec_to_main.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_KYC"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_3", "AdHoc_E1_Wrapper", "AdHocSubProcess_KYC", None),
            ("Flow_4", "AdHocSubProcess_KYC", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc26_variable_exec_to_main_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.enrichedProfile"],
                "task_ids": ["UserTask_Input", "QHanaTask_EnrichProfile"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_E1_end"],
            }
        ],
    },
    "tc27_variable_main_to_exec.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Clinical"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "AdHocSubProcess_Clinical", None),
            ("Flow_3", "AdHocSubProcess_Clinical", "AdHoc_E2_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E2_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc27_variable_main_to_exec_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.patientId"],
                "task_ids": ["UserTask_Input"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc27_variable_main_to_exec_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.scanData"],
                "outputs": [],
                "task_ids": ["QHanaTask_AnalyzeScans"],
                "flow_ids": ["Flow_E2_start", "Flow_4"],
            },
        ],
    },
    "tc28_data_object_cross_boundary.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Review"),
            ("endEvent", "EndEvent_1"),
            ("dataObject", "DataObject_Doc"),
            ("dataObjectReference", "DataObjectReference_Doc"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_3", "AdHoc_E1_Wrapper", "AdHocSubProcess_Review", None),
            ("Flow_4", "AdHocSubProcess_Review", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc28_data_object_cross_boundary_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.contractDoc"],
                "task_ids": ["UserTask_Input", "QHanaTask_DraftContract"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_E1_end"],
            }
        ],
    },
    "tc29_association_cleanup.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Audit"),
            ("endEvent", "EndEvent_1"),
            ("textAnnotation", "TextAnnotation_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_3", "AdHoc_E1_Wrapper", "AdHocSubProcess_Audit", None),
            ("Flow_4", "AdHocSubProcess_Audit", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc29_association_cleanup_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.filteredLogs"],
                "task_ids": ["UserTask_Input", "QHanaTask_FilterLogs"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_E1_end"],
            }
        ],
    },
    "tc30_send_task.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc30_send_task_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "QHanaTask_PrepareMessage",
                    "SendTask_Notify",
                    "QHanaTask_Finalize",
                ],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3", "Flow_4"],
            }
        ],
    },
    "tc31_receive_task.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc31_receive_task_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "QHanaTask_Request",
                    "ReceiveTask_Wait",
                    "QHanaTask_ProcessResponse",
                ],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3", "Flow_4"],
            }
        ],
    },
    "tc32_message_inside_exec_vs_main.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_1"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "AdHocSubProcess_1", None),
            ("Flow_3", "AdHocSubProcess_1", "AdHoc_E2_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E2_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc32_message_inside_exec_vs_main_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.updatePayload"],
                "task_ids": ["QHanaTask_Prepare"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc32_message_inside_exec_vs_main_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.approvedUpdate"],
                "outputs": [],
                "task_ids": ["SendTask_Update", "QHanaTask_Finalize"],
                "flow_ids": ["Flow_E2_start", "Flow_4", "Flow_5"],
            },
        ],
    },
    "tc33_pool_lane_preservation.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Review"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "AdHocSubProcess_Review", None),
            ("Flow_3", "AdHocSubProcess_Review", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc33_pool_lane_preservation_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.configResult"],
                "task_ids": ["QHanaTask_Config"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            }
        ],
    },
    "tc34_lane_partial_removal.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_ManualAction"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "AdHocSubProcess_ManualAction", None),
            ("Flow_3", "AdHocSubProcess_ManualAction", "AdHoc_E2_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E2_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc34_lane_partial_removal_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["QHanaTask_AutomatedPrep"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc34_lane_partial_removal_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["QHanaTask_AutomatedCleanup"],
                "flow_ids": ["Flow_E2_start", "Flow_4"],
            },
        ],
    },
    "tc35_multiple_pools.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Fulfillment"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "AdHocSubProcess_Fulfillment", None),
            ("Flow_3", "AdHocSubProcess_Fulfillment", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc35_multiple_pools_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["QHanaTask_ReceiveOrder"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            }
        ],
    },
    "tc36_di_cleanup.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_1"),
            ("adHocSubProcess[wrapper=E2]", "AdHoc_E2_Wrapper"),
            ("endEvent", "EndEvent_Main"),
            ("textAnnotation", "TextAnnotation_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "AdHocSubProcess_1", None),
            ("Flow_3", "AdHocSubProcess_1", "AdHoc_E2_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E2_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc36_di_cleanup_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": ["qoutput.prepared"],
                "task_ids": ["QHanaTask_Prepare"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            },
            {
                "fragment_id": "E2",
                "process_id": "tc36_di_cleanup_E2",
                "wrapper_id": "AdHoc_E2_Wrapper",
                "inputs": ["qoutput.approved"],
                "outputs": [],
                "task_ids": ["QHanaTask_Finalize"],
                "flow_ids": ["Flow_E2_start", "Flow_4"],
            },
        ],
    },
    "tc37_extension_preservation.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc37_extension_preservation_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["QHanaTask_Extended", "ServiceTask_External"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3"],
            }
        ],
    },
    "tc38_custom_attributes.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc38_custom_attributes_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["QHanaTask_Metadata", "UserTask_Confirm"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3"],
            }
        ],
    },
    "tc39_compressed_flow_condition.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("exclusiveGateway", "Gateway_Split"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Manual"),
            ("exclusiveGateway", "Gateway_Merge"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_Cond", "Gateway_Split", "AdHoc_E1_Wrapper", "${urgent == true}"),
            ("Flow_1", "StartEvent_1", "Gateway_Split", None),
            ("Flow_Default", "Gateway_Split", "AdHocSubProcess_Manual", None),
            ("Flow_EndA", "AdHoc_E1_Wrapper", "Gateway_Merge", None),
            ("Flow_EndB", "AdHocSubProcess_Manual", "Gateway_Merge", None),
            ("Flow_Final", "Gateway_Merge", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc39_compressed_flow_condition_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["QHanaTask_Exec1", "QHanaTask_Exec2"],
                "flow_ids": ["Flow_E1_start", "Flow_Mid", "Flow_E1_end"],
            }
        ],
    },
    "tc40_default_flow_remap.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("exclusiveGateway", "Gateway_Split"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHocSubProcess_Manual"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "Gateway_Split", None),
            ("Flow_Target_Exec", "Gateway_Split", "AdHoc_E1_Wrapper", None),
            (
                "Flow_Target_AdHoc",
                "Gateway_Split",
                "AdHocSubProcess_Manual",
                "${quality == 'fail'}",
            ),
            ("Flow_to_Main_End_0", "AdHocSubProcess_Manual", "EndEvent_Main", None),
            ("Flow_to_Main_End_1", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc40_default_flow_remap_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["QHanaTask_Automated"],
                "flow_ids": ["Flow_E1_start", "Flow_EndA"],
            }
        ],
    },
    "tc41_multiple_flows_same_src_target.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc41_multiple_flows_same_src_target_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["Gateway_Split", "QHanaTask_Target"],
                "flow_ids": ["Flow_E1_start", "Flow_Path_A", "Flow_Path_B", "Flow_End"],
            }
        ],
    },
    "tc42_no_dangling_nodes.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("exclusiveGateway", "Gateway_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHoc_1"),
            ("exclusiveGateway", "Gateway_2"),
            ("endEvent", "EndEvent_1"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "Gateway_1", None),
            ("Flow_2", "Gateway_1", "AdHoc_E1_Wrapper", "${route == 'auto'}"),
            ("Flow_3", "Gateway_1", "AdHoc_1", None),
            ("Flow_4", "AdHoc_E1_Wrapper", "Gateway_2", None),
            ("Flow_5", "AdHoc_1", "Gateway_2", None),
            ("Flow_6", "Gateway_2", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc42_no_dangling_nodes_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["Task_Exec"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            }
        ],
    },
    "tc43_start_to_end_path.bpmn": {
        "nsup": "Ad-hoc subprocess 'AdHoc_1' contains non-QHAna task 'Setup_Task' "
        "(type='userTask'). All tasks inside an ad-hoc must be QHAna-recognized for "
        "UI template generation."
    },
    "tc44_no_orphan_elements.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc44_no_orphan_elements_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["Task_Exec", "BoundaryEvent_1"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3"],
            }
        ],
    },
    "tc45_wu_palmer_partial.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "Activity_AdHocCluster"),
            ("endEvent", "Event_0ck8ig8"),
        ],
        "main_flows": [
            ("Flow_05mvkgh", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_To_AdHoc", "AdHoc_E1_Wrapper", "Activity_AdHocCluster", None),
            ("Flow_To_End", "Activity_AdHocCluster", "Event_0ck8ig8", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "Wu_Palmer_Mixed_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [
                    "start_attributes",
                    "start_entitiesMetadataUrl",
                    "start_entitiesUrl",
                    "start_mdsDimensions",
                    "start_taxonomiesZipUrl",
                ],
                "outputs": ["return.qoutput.mds"],
                "task_ids": [
                    "Activity_1do8hxs",
                    "Activity_0tfwzt0",
                    "Activity_1nnwor0",
                    "Activity_0s7n3hs",
                    "Activity_0itt0yf",
                ],
                "flow_ids": [
                    "Flow_E1_start",
                    "Flow_13r8zml",
                    "Flow_0sl6imb",
                    "Flow_0w1duld",
                    "Flow_0texvqp",
                    "Flow_E1_end",
                ],
            }
        ],
    },
    "tc46_loop_markers.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc46_loop_markers_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "ServiceTask_Prepare",
                    "ServiceTask_Loop",
                    "ServiceTask_Aggregate",
                ],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3", "Flow_4"],
            }
        ],
    },
    "tc47_subprocess_all_extractable.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc47_subprocess_all_extractable_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["SubProcess_Extractable"],
                "flow_ids": ["Flow_E1_start", "Flow_end"],
            }
        ],
    },
    "tc48_subprocess_all_main_side.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc48_subprocess_all_main_side_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["SubProcess_MainSide"],
                "flow_ids": ["Flow_E1_start", "Flow_end"],
            }
        ],
    },
    "tc49_subprocess_mixed_unsupported.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc49_subprocess_mixed_unsupported_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["SubProcess_Mixed"],
                "flow_ids": ["Flow_E1_start", "Flow_end"],
            }
        ],
    },
    "tc50_subprocess_with_boundary_event.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc50_subprocess_with_boundary_event_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["SubProcess_WithBoundary"],
                "flow_ids": ["Flow_E1_start", "Flow_end"],
            }
        ],
    },
    "tc51_multiple_end_events.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc51_multiple_end_events_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "ServiceTask_Check",
                    "Gateway_Split",
                    "ServiceTask_Approve",
                    "ServiceTask_Reject",
                ],
                "flow_ids": [
                    "Flow_E1_start",
                    "Flow_2",
                    "Flow_Approve",
                    "Flow_Reject",
                    "Flow_To_Approved",
                    "Flow_To_Rejected",
                ],
            }
        ],
    },
    "tc52_parallel_end_events.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc52_parallel_end_events_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "Gateway_Fork",
                    "ServiceTask_ArchiveA",
                    "ServiceTask_ArchiveB",
                ],
                "flow_ids": [
                    "Flow_E1_start",
                    "Flow_A",
                    "Flow_B",
                    "Flow_End_A",
                    "Flow_End_B",
                ],
            }
        ],
    },
    "tc53_intermediate_events.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc53_intermediate_events_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "ServiceTask_Before",
                    "Catch_Timer",
                    "ServiceTask_After",
                    "Throw_Signal",
                ],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_3", "Flow_4", "Flow_5"],
            }
        ],
    },
    "tc54_specialized_end_events.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc54_specialized_end_events_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["ServiceTask_Run", "Gateway_Split"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_Ok", "Flow_Fail"],
            }
        ],
    },
    "tc55_data_store.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
            ("dataStoreReference", "DataStoreRef_Results"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc55_data_store_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["ServiceTask_Persist"],
                "flow_ids": ["Flow_E1_start", "Flow_2"],
            }
        ],
    },
    "tc56_group.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("adHocSubProcess[original]", "AdHoc_Analyze"),
            ("endEvent", "EndEvent_1"),
            ("group", "Group_Setup"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_2", "AdHoc_E1_Wrapper", "AdHoc_Analyze", None),
            ("Flow_3", "AdHoc_Analyze", "EndEvent_1", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc56_group_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["ServiceTask_Prep"],
                "flow_ids": ["Flow_E1_start", "Flow_E1_end"],
            }
        ],
    },
    "tc57_escalation_end.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc57_escalation_end_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["ServiceTask_Check", "Gateway_Severity"],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_Urgent", "Flow_Normal"],
            }
        ],
    },
    "tc58_signal_end.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc58_signal_end_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": ["ServiceTask_ProcessBatch"],
                "flow_ids": ["Flow_E1_start", "Flow_2"],
            }
        ],
    },
    "tc59_non_interrupting_boundary.bpmn": {
        "main_nodes": [
            ("startEvent", "StartEvent_1"),
            ("adHocSubProcess[wrapper=E1]", "AdHoc_E1_Wrapper"),
            ("endEvent", "EndEvent_Main"),
        ],
        "main_flows": [
            ("Flow_1", "StartEvent_1", "AdHoc_E1_Wrapper", None),
            ("Flow_to_Main_End", "AdHoc_E1_Wrapper", "EndEvent_Main", None),
        ],
        "fragments": [
            {
                "fragment_id": "E1",
                "process_id": "tc59_non_interrupting_boundary_E1",
                "wrapper_id": "AdHoc_E1_Wrapper",
                "inputs": [],
                "outputs": [],
                "task_ids": [
                    "ServiceTask_LongJob",
                    "Boundary_Timer_NonInt",
                    "ServiceTask_Notify",
                ],
                "flow_ids": ["Flow_E1_start", "Flow_2", "Flow_Notify", "Flow_NotifyEnd"],
            }
        ],
    },
}
