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

var save_table_value = document.getElementById("save_table");

var attribute_to_id_vis = document.getElementById("id_attribute").parentNode.parentNode;


function save_table_change() {
    attribute_to_id_vis.style.display = "none";
    if (save_table_value.checked === true){
        attribute_to_id_vis.style.display = "block"
    }
}

save_table_change();

save_table_value.addEventListener("change", save_table_change);
