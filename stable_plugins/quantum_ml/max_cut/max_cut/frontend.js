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

var max_cut_enum_val = document.getElementById("max_cut_enum");
var max_trials_val = document.getElementById("max_trials");
var reps_val = document.getElementById("reps");
var backend_val = document.getElementById("backend");
var shots_val = document.getElementById("shots");
var ibmq_custom_backend_val = document.getElementById("ibmq_custom_backend");
var ibmq_token_val = document.getElementById("ibmq_token");

var optimizer_dis = document.getElementById("optimizer").parentNode.parentNode;
var max_trials_dis = max_trials_val.parentNode.parentNode;
var reps_dis = reps_val.parentNode.parentNode;
var entanglement_pattern_enum_dis = document.getElementById("entanglement_pattern_enum").parentNode.parentNode;
var backend_dis = backend_val.parentNode.parentNode;
var shots_dis = shots_val.parentNode.parentNode;
var ibmq_custom_backend_dis = ibmq_custom_backend_val.parentNode.parentNode;
var ibmq_token_dis = ibmq_token_val.parentNode.parentNode;


function default_values() {
    max_trials_val.value = 10;
    reps_val.value = 4;
    shots_val.value = 1000;
}


function max_cut_enum_change() {
    optimizer_dis.style.display = "none";
    max_trials_dis.style.display = "none";
    reps_dis.style.display = "none";
    entanglement_pattern_enum_dis.style.display = "none";
    backend_dis.style.display = "none";
    shots_dis.style.display = "none";
    ibmq_custom_backend_dis.style.display = "none";
    ibmq_token_dis.style.display = "none";
    if (max_cut_enum_val.value === "vqe") {
        optimizer_dis.style.display = "block";
        max_trials_dis.style.display = "block";
        reps_dis.style.display = "block";
        entanglement_pattern_enum_dis.style.display = "block";
        backend_dis.style.display = "block";
        shots_dis.style.display = "block";
        backend_enum_change();
    }
    else {
        // Sets default value for hidden variables, avoiding validation errors
        default_values();
    }
}


function backend_enum_change() {
    ibmq_custom_backend_dis.style.display = "none";
    ibmq_token_dis.style.display = "none";
    if (backend_val.value === "custom_ibmq") {
        ibmq_custom_backend_dis.style.display = "block";
        ibmq_token_dis.style.display = "block";
    }
    else if (backend_val.value.startsWith("ibmq")) {
        ibmq_token_dis.style.display = "block";
    }
}

max_cut_enum_val.addEventListener("change", max_cut_enum_change);
max_cut_enum_change();
backend_val.addEventListener("change", backend_enum_change);
backend_enum_change();