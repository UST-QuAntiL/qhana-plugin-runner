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
var qknn_type_value = document.getElementById("variant");
var k_value = document.getElementById("k");
var exp_itr_value = document.getElementById("exp_itr");
var slack_value = document.getElementById("slack");

// If a vairable ends with _vis, it's the parentNode's parentNode and we can set the visibility
var k_vis = k_value.parentNode.parentNode;
var exp_itr_vis = exp_itr_value.parentNode.parentNode;
var slack_vis = slack_value.parentNode.parentNode;


function set_default_values() {
    exp_itr_value.value = "10";
    k_value = 1;
    slack_value = 0.05;
}


function hide_all() {
    exp_itr_vis.style.display = 'none';
    k_vis.style.display = 'none';
    slack_vis.style.display = 'none';
}


function show_basheer_hamming() {
    exp_itr_vis.style.display = 'block';
    slack_vis.style.display = 'block';
}


function qknn_type_change(reset_values=true) {
    hide_all();
    if (reset_values) {
        set_default_values();
    }
    if (qknn_type_value.value !== "schuld_qknn") {
        k_vis.style.display = 'block';
    }
    if (qknn_type_value.value === "basheer_hamming_qknn") {
        show_basheer_hamming();
    }
}

qknn_type_change(false);

qknn_type_value.addEventListener("change", qknn_type_change);
