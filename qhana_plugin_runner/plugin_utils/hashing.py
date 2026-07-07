# Copyright 2026 QHAna plugin runner contributors.
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

import muid


def get_readable_hash(s: str) -> str:
    """Hash a string into a short human-readable identifier.

    Args:
        s (str): the string to hash

    Returns:
        str: a hyphen-separated memorable unique id
    """
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")
