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

var noise_vis = document.getElementById("noise").parentNode.parentNode;
var turns_vis = document.getElementById("turns").parentNode.parentNode;
var dataset_type = document.getElementById("dataset_type");

function dataset_type_change() {
    console.log("data_type_change");
    noise_vis.style.display = 'none';
    turns_vis.style.display = 'none';
    if (dataset_type.value === "two_spirals") {
        noise_vis.style.display = 'block';
        turns_vis.style.display = 'block';
    }
}

dataset_type.addEventListener("change", dataset_type_change);
