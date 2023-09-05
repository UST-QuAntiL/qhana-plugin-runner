// Copyright 2023 QHAna plugin runner contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

var ibmq_token = document.getElementById("ibmq_token").parentNode.parentNode;
var custom_backend = document.getElementById("custom_backend").parentNode.parentNode;

backend_type = document.getElementById("backend");

function backend_type_change() {
    ibmq_token.style.display = 'none';
    custom_backend.style.display = 'none';
    if (backend_type.value.startsWith("ibmq")) {
        ibmq_token.style.display = 'block';
    } else if (backend_type.value === "custom_ibmq") {
        ibmq_token.style.display = 'block';
        custom_backend.style.display = 'block';
    }
}

backend_type_change();

// react to changes
backend_type.addEventListener("change", backend_type_change);
