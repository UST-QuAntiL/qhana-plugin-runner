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

import marshmallow as ma


def validate_floats_seperated_by_comma(value: str):
    """
    Checks if a given string consits of floats separated by commas and each float is greater than 0. Whitespaces are
    ignored. If this is not the case, then it throws a ValidationError.
    """
    if value != "":
        value = value.replace(" ", "")
        value = value.split(",")
        for entry in value:
            if not entry.isdigit():
                raise ma.ValidationError("Not numbers separated by commas.")
            entry = int(entry)
            if entry <= 0:
                raise ma.ValidationError("Numbers must be greater than 0.")
