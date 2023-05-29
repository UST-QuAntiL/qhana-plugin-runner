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

// Long block of just gathering elements
// If a variable ends with _value, it's the element itself and the value can be retrieved or set
var backend_value = document.getElementById("backend");

// If a vairable ends with _vis, it's the parentNode's parentNode and we can set the visibility
var ibmq_token_vis = document.getElementById("ibmq_token").parentNode.parentNode;
var custom_backend_vis = document.getElementById("custom_backend").parentNode.parentNode;


function backend_change() {
    ibmq_token_vis.style.display = 'none';
    custom_backend_vis.style.display = 'none';
    if (backend_value.value.startsWith("ibmq") || backend_value.value === "aer_qasm_simulator") {
        ibmq_token_vis.style.display = 'block';
        custom_backend_vis.style.display = 'none';
    } else if (backend_value.value === "custom_ibmq") {
        ibmq_token_vis.style.display = 'block';
        custom_backend_vis.style.display = 'block';
    }
}

backend_change();

backend_value.addEventListener("change", backend_change);
