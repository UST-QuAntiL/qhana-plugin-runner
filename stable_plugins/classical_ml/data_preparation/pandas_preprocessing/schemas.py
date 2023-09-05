# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from marshmallow import post_load, validate
import marshmallow as ma
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from qhana_plugin_runner.api import EnumField

from dataclasses import dataclass
from .backend.pandas_preprocessing import (
    PreprocessingEnum,
    AxisEnum,
    KeepEnum,
    PositionEnum,
    CaseEnum,
)

from celery.utils.log import get_task_logger


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class FirstInputParameters:
    file_url: str

    def __str__(self):
        return str(self.__dict__)


class FirstInputParametersSchema(FrontendFormBaseSchema):
    file_url = FileUrl(
        required=True,
        allow_none=False,
        metadata={
            "label": "File URL",
            "description": "A url to the file.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> FirstInputParameters:
        return FirstInputParameters(**data)


@dataclass(repr=False)
class SecondInputParameters:
    preprocessing_enum: PreprocessingEnum
    axis: AxisEnum
    threshold: int
    subset: str
    fill_value: str
    keep: KeepEnum
    by: str
    position: PositionEnum
    characters: str
    column: str
    new_columns: str
    substring: str
    new_str: str
    case: CaseEnum
    ignore_index: bool = False
    ascending: bool = False
    remove_column: bool = False

    def __str__(self):
        return str(self.__dict__)


class SecondInputParametersSchema(FrontendFormBaseSchema):
    preprocessing_enum = EnumField(
        PreprocessingEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Preprocessing Type",
            "description": """Type of preprocessing the file will undergo.\n
- do nothing: Does nothing to the file, in case the user accidentally clicked the button `add step`, instead of `done`.\n
- drop na: Removes rows or columns with a certain amount of missing values.\n
- fill na: Fills in empty entries with a given value.\n
- drop duplicates: Removes duplicate rows. May only consider certain columns, to determine if two rows are duplicates.\n
- sort values: Sorts rows by their values in a certain column.\n
- strip entries: Strips entries in the specified columns by the given characters. The beginning, the end or both of an entry may be stripped.\n
- split column: Splits a column into, given a substring to split entries by.\n
- replace: Replaces all occurrences of a string in all entries with another string.\n
- string case: Transforms entries into upper, lower or title case.""",
            "input_type": "select",
        },
    )
    axis = EnumField(
        AxisEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Axis",
            "description": "Determine if rows or columns which contain missing values are dropped.",
            "input_type": "select",
        },
    )
    threshold = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Threshold",
            "description": "Requires that many non-NA values. Cannot be combined with how. If left empty, then all values may not be NA.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=True),
    )
    subset = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Subset",
            "description": "To be set by javascript.",
            "input_type": "text",
        },
    )
    fill_value = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Value",
            "description": "The value to fill every na with.",
            "input_type": "text",
        },
    )
    keep = EnumField(
        KeepEnum,
        required=False,
        allow_none=True,
        metadata={
            "label": "Keep",
            "description": """Determines which duplicates (if any) to keep.\n
            - first: Drop duplicates except for the first occurrence.\n
            - last: Drop duplicates except for the last occurrence.\n
            - none: Drop all duplicates.""",
            "input_type": "select",
        },
    )
    ignore_index = ma.fields.Boolean(
        required=False,
        allow_none=True,
        metadata={
            "label": "Ignore index",
            "description": "If unchecked, each element keeps its current row index. If checked, the resulting axis will be relabeled 0, 1, …, n - 1.",
            "input_type": "checkbox",
        },
    )
    by = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "To be set by javascript.",
            "description": "To be set by javascript.",
            "input_type": "text",
        },
    )
    ascending = ma.fields.Boolean(
        required=False,
        allow_none=True,
        metadata={
            "label": "Sorting order",
            "description": "Sorts by ascending order, if checked and descending order, otherwise.",
            "input_type": "checkbox",
        },
    )
    position = EnumField(
        PositionEnum,
        required=False,
        allow_none=True,
        metadata={
            "label": "Position",
            "description": """Determines where to strip an entry of the given characters.\n
            - front: Strips the given characters from the beginning of the entry.\n
            - end: Strips the given characters from the end of the entry.\n
            - both: Strips the given characters from both the beginning and the end of the entry.""",
            "input_type": "select",
        },
    )
    characters = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Characters",
            "description": "Each character entered will be stripped from an entry.",
            "input_type": "text",
        },
    )
    column = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Column",
            "description": "Column that will be split.",
            "input_type": "text",
        },
    )
    new_columns = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "New columns",
            "description": "Names for the new columns separated by a comma.",
            "input_type": "text",
        },
    )
    remove_column = ma.fields.Boolean(
        required=False,
        allow_none=True,
        metadata={
            "label": "Remove old column",
            "description": "If checked, the old column will be removed.",
            "input_type": "checkbox",
        },
    )
    substring = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Substring",
            "description": "This is the substring that will be replaced by a new string in each entry.",
            "input_type": "text",
        },
    )
    new_str = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "New string",
            "description": "This is the string that the substring will be replaced with.",
            "input_type": "text",
        },
    )
    case = EnumField(
        CaseEnum,
        required=False,
        allow_none=True,
        metadata={
            "label": "Case",
            "description": """Determines the case that will be applied to each entry. Possible cases are:\n
            - upper: sTrIng becomes STRING\n
            - lower: sTrIng becomes string\n
            - title: sTrIng becomes String""",
            "input_type": "select",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> SecondInputParameters:
        return SecondInputParameters(**data)
