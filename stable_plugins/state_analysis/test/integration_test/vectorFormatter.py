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


def format_complex_vectors(vectors):
    formatted_vectors = []
    for vec in vectors:
        fv = []
        for cnum in vec:
            fv.append([cnum.real, cnum.imag])
        formatted_vectors.append(fv)

    return formatted_vectors
