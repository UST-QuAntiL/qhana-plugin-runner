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

// Precomputed Kernel Stuff
var train_kernel_url_dis = document.getElementById("train_kernel_url").parentNode.parentNode;
var test_kernel_url_dis = document.getElementById("test_kernel_url").parentNode.parentNode;
var train_points_url_dis = document.getElementById("train_points_url").parentNode.parentNode;
var test_points_url_dis = document.getElementById("test_points_url").parentNode.parentNode;

// Polynomial Kernel Stuff
var degree_dis = document.getElementById("degree").parentNode.parentNode;

// Quantum Kernel Stuff
var data_maps_enum_dis = document.getElementById("data_maps_enum").parentNode.parentNode;
var entanglement_pattern_dis = document.getElementById("entanglement_pattern").parentNode.parentNode;
var paulis_dis = document.getElementById("paulis").parentNode.parentNode;
var reps_dis = document.getElementById("reps").parentNode.parentNode;
var shots_dis = document.getElementById("shots").parentNode.parentNode;
var backend_dis = document.getElementById("backend").parentNode.parentNode;
var ibmq_token_dis = document.getElementById("ibmq_token").parentNode.parentNode;
var custom_backend_dis = document.getElementById("custom_backend").parentNode.parentNode;

var kernel_enum_val = document.getElementById("kernel_enum");


function kernel_change() {
    train_kernel_url_dis.style.display = 'none';
    test_kernel_url_dis.style.display = 'none';
    train_points_url_dis.style.display = 'block';
    test_points_url_dis.style.display = 'block';
    degree_dis.style.display = 'none';
    data_maps_enum_dis.style.display = 'none';
    entanglement_pattern_dis.style.display = 'none';
    paulis_dis.style.display = 'none';
    reps_dis.style.display = 'none';
    shots_dis.style.display = 'none';
    backend_dis.style.display = 'none';
    ibmq_token_dis.style.display = 'none';
    custom_backend_dis.style.display = 'none';

    if (kernel_enum_val.value === "precomputed") {
        train_kernel_url_dis.style.display = 'block';
        test_kernel_url_dis.style.display = 'block';
        train_points_url_dis.style.display = 'none';
        test_points_url_dis.style.display = 'none';
    }
    else if (kernel_enum_val.value === "poly") {
        degree_dis.style.display = 'block';
    }
    else if (
        kernel_enum_val.value === "z_kernel"
        || kernel_enum_val.value === "zz_kernel"
        || kernel_enum_val.value === "pauli_kernel"
    ) {
        data_maps_enum_dis.style.display = 'block';
        entanglement_pattern_dis.style.display = 'block';
        reps_dis.style.display = 'block';
        shots_dis.style.display = 'block';
        backend_dis.style.display = 'block';
        ibmq_token_dis.style.display = 'block';
        custom_backend_dis.style.display = 'block';

        if (kernel_enum_val.value === "pauli_kernel") {
            paulis_dis.style.display = 'block';
        }
    }
}

kernel_change();

kernel_enum_val.addEventListener("change", kernel_change);
