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

"""Utilities for unit tests."""

from typing import Any, Sequence


def assert_sequence_equals(expected: Sequence[Any], actual: Sequence[Any]):
    """Assert that two sequences contain matching elements."""
    assert len(actual) == len(
        expected
    ), f"Sequences have different sizes, expected length {len(expected)} but got legth {len(actual)}"
    for index, pair in enumerate(zip(actual, expected)):
        actual_item, expected_item = pair
        assert (
            actual_item == expected_item
        ), f"Pair {index} is not equal. Expected {expected_item} but got {actual_item}"


def assert_sequence_partial_equals(
    expected: Sequence[Any], actual: Sequence[Any], attributes_to_test: Sequence[str]
):
    """Assert that the elements in a sequence match in all attributes defined by ``attributes_to_test``.

    The elements in the list can be dicts or namedtuples.
    """
    assert len(actual) == len(
        expected
    ), f"Sequences have different sizes, expected length {len(expected)} but got legth {len(actual)}"
    for index, pair in enumerate(zip(actual, expected)):
        actual_item, expected_item = pair
        for attr in attributes_to_test:
            if isinstance(actual_item, dict):
                actual_value = actual_item.get(attr)
            else:
                actual_value = getattr(actual_item, attr)
            if isinstance(expected_item, dict):
                expected_value = expected_item.get(attr)
            else:
                expected_value = getattr(expected_item, attr)
            assert (
                expected_value == actual_value
            ), f"Attribute '{attr}' of pair {index} is not equal ({expected_value}!={actual_value}). Expected {expected_item} but got {actual_item}"
