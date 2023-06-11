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

var db_enum_value = document.getElementById("db_enum");

var db_host_vis = document.getElementById("db_host").parentNode.parentNode;
var db_port_vis = document.getElementById("db_port").parentNode.parentNode;
var db_user_vis = document.getElementById("db_user").parentNode.parentNode;
var db_password_vis = document.getElementById("db_password").parentNode.parentNode;

function db_enum_change() {
    db_host_vis.style.display = "none";
    db_port_vis.style.display = "none";
    db_user_vis.style.display = "none";
    db_password_vis.style.display = "none";
    if (db_enum_value.value === "auto" || db_enum_value.value === "postgresql") {
        db_host_vis.style.display = "block";
        db_port_vis.style.display = "block";
        db_user_vis.style.display = "block";
        db_password_vis.style.display = "block";
    } else if (db_enum_value.value === "mysql") {
        db_host_vis.style.display = "block";
        db_port_vis.style.display = "block";
        db_user_vis.style.display = "block";
        db_password_vis.style.display = "block";
    }
}

db_enum_change();

db_enum_value.addEventListener("change", db_enum_change);
