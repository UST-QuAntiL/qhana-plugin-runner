import re

def get_array_from_table_str(s):
    """
    Input:
        "| Data Type | Content Type | Required |
        |-----------|--------------| :------: |
        |example-data-type|example-content-type|✓|"
    Output:
        [
            [
                " Data Type ",
                " Content Type ",
                " Required "
            ],
            [
                "-----------",
                "--------------",
                " :------: "
            ],
            [
                "example-data-type",
                "example-content-type",
                "✓"
            ]
        ]
    """
    return [x[1:-1].split("|") for x in s.split("\n")]


def process_math_to_array(s):
    """
    Input:
        Either no regex match, or a match with group(1) =
        "| Data Type | Content Type | Required |
        |-----------|--------------| :------: |
        |...|...|...|
        |example-data-type|example-content-type|✓|"
    Output:
        [
            [
                "example-data-type",
                ["example-content-type"],
                True
            ]
        ]
    """
    if s is None:
        return []

    array = get_array_from_table_str(s.group(1))[2:]
    return [[x[0], x[1].split(", "), x[2] == '✓'] for x in array]

def get_plugin_information(plugin_description_file):
    """
    Extracts the plugins information from the all-plugins.md in the qhana-plugin-runner documentation

    Input: file with contents of form:
        ...
        ## Plugins
        ...
        ### First Plugin (@v1.0.0)
        ...
        **Inputs:**

        | Data Type | Content Type | Required |
        |-----------|--------------| :------: |
        |...|...|...|
        |example-data-type|example-content-type|✓|


        **Outputs:**

        | Data Type | Content Type | Required |
        |-----------|--------------| :------: |
        |...|...|...|
        |example-data-type|example-content-type|╳|


        ...
        ### Second Plugin (@v2.0.0)
        ...
    Output:
        {
            'First Plugin (@v1.0.0)': {
                'input': [
                    [...],
                    [
                        'example-data-type',
                        ['example-content-type'],
                        True (depending on whether this input is required)
                    ]
                ],
                'output': [
                    [...],
                    [
                        'example-data-type',
                        ['example-content-type'],
                        False (depending on whether this output is required)
                    ]
                ]
            },
            'Second Plugin (@v2.0.0)': ...
        }
    """
    with open(plugin_description_file, "r", encoding='utf-8') as file:
        raw_data = "".join(file.readlines())

    # Extracting the input, output information for each plugin
    plugin_strs = raw_data.split("## Plugins")[1].split("### ")[1:]

    # counter_proc = 0
    # counter_vis = 0
    # counter_preproc = 0

    plugins = {}
    for plugin_str in plugin_strs:
        # if "processing" in plugin_str:
        #     counter_proc += 1
        # if "visualization" in plugin_str:
        #     counter_vis += 1
        # if "preprocessing" in plugin_str:
        #     counter_preproc += 1
        name = re.search("(^.*)\n", plugin_str).group(1)
        
        input_str_match = re.search("[*]{2}Inputs:[*]{2}\n\n(\|(.|\n\|)*)\n\n", plugin_str)
        input = process_math_to_array(input_str_match)

        output_str_match = re.search("[*]{2}Outputs:[*]{2}\n\n(\|(.|\n\|)*)\n\n", plugin_str)
        output = process_math_to_array(output_str_match)

        plugins[name] = {"input": input, "output": output}

    # print(counter_proc, counter_vis, counter_preproc)
    return plugins
