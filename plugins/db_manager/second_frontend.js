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


const font_size = 13. + 1./3.

let check_box_list;
let table_select;

const save_table_value = document.getElementById("save_table");
const columns_list_value = document.getElementById("columns_list");
const table_name_value = document.getElementById("table_name");
const tables_and_columns = JSON.parse(document.getElementById("additional_info").textContent);


const attribute_to_id_vis = document.getElementById("id_attribute").parentNode.parentNode;


function create_checkbox(name, check_status=false){
    let label_element = document.createElement("label");
    label_element.setAttribute("for", name);
    label_element.style.cssText = "display: block; color: WindowText; background-color: Window; margin: 0; padding: 0; width: 100%;";

    let input_element = document.createElement("input");
    input_element.setAttribute("type", "checkbox");
    input_element.setAttribute("name", name);
    input_element.setAttribute("id", name+"_box");
    input_element.checked = check_status;
    // input_element.addEventListener("change", append_value_to_str_list, false);
    label_element.appendChild(input_element);

    let text = document.createTextNode(name);
    label_element.appendChild(text);


    label_element.style.font = font_size+"px DejaVu,Sans";

    return label_element;
}


function update_check_all_box_indeterminate(event){
    let check_box = event.target;
    let check_all_box = document.getElementById("select all_box");

    if (!check_all_box.indeterminate) {
        check_all_box.indeterminate = (check_box.checked !== check_all_box.checked);
    }
}


function create_checkbox_for_list(name, check_status=false) {
    let list_element = document.createElement("li");
    list_element.style.cssText = "margin: 0; padding: 0;";
    let check_box = create_checkbox(name, check_status);
    check_box.children[0].addEventListener("change", update_check_all_box_indeterminate);
    list_element.appendChild(check_box);

    return list_element;
}


function set_all_check_box_list(checked_values){
    let check_box_list = document.getElementById("check_box_list");
    //iterate through li elements
    for (let i = 0; i < check_box_list.children.length; i++) {
        // Set checked status
        check_box_list.children[i].children[0].children[0].checked = checked_values;
    }
}


function check_all_box_change(event) {
    set_all_check_box_list(event.target.checked);
}


function init_check_all_box() {
    if (document.getElementById("select all_id") === null) {
        let check_all_box = create_checkbox("select all");
        // check_all_box.children[0].indeterminate = true;
        check_all_box.children[0].addEventListener("change", check_all_box_change);
        check_all_box.outerHTML = "<br />";

        columns_list_value.parentNode.appendChild(check_all_box);
    }
}

function create_check_box_list() {
    if (document.getElementById("check_box_list") !== null) {
        document.getElementById("check_box_list").remove();
    }
    check_box_list = document.createElement("ul");
    check_box_list.id = "check_box_list";
    check_box_list.className = "qhana-form-input";
    check_box_list.style.cssText = "height: 100px; overflow: auto; max-width: 25em; border: 1px solid #8f8f9d; list-style-type: none; margin: 0; padding: 0.3em; flex-grow: 1; overflow-x: hidden;";
    // dummy_field_value.style.cssText = "height: 100px; overflow: auto; max-width: 25em; border: 1px solid #8f8f9d; list-style-type: none; margin: 0; padding: 0.3em; flex-grow: 1; overflow-x: hidden;";

    let current_table = table_select.value;
    let columns = tables_and_columns[current_table];

    let check_status = columns_list_value.value.split(",");
    for (let i = 0; i < columns.length; i++) {
        check_box_list.appendChild(create_checkbox_for_list(columns[i].toString(), check_status.includes(columns[i].toString())));
        // dummy_field_value.appendChild(create_checkbox_for_list(i.toString()));
    }

    columns_list_value.parentNode.appendChild(check_box_list);
}


function write_checkbox_content_to_str_list(){
    console.log("on click duty!");
    // let check_box_list = document.getElementById("check_box_list");
    let result = [];
    //iterate through li elements
    for (let i = 0; i < check_box_list.children.length; i++) {
        // get checkbox element
        let checkbox_element = check_box_list.children[i].children[0].children[0];

        console.log(checkbox_element.checked);
        if (checkbox_element.checked){
            result.push(checkbox_element.name);
        }
    }
    console.log("result: [" + result + "]");
    columns_list_value.value = result;
}


function init_table_name_value(){
    table_name_value.style.display = "none";
    let table_select = document.createElement("select");
    table_select.setAttribute("id", "table_select");
    table_select.setAttribute("class", "qhana-form-input");

    Object.keys(tables_and_columns).forEach(function(key) {
        let table_option = document.createElement("option")
        table_option.setAttribute("value", key);
        table_option.innerText = key.toString();
        table_select.appendChild(table_option);
    });

    table_name_value.parentNode.appendChild(table_select);
    return table_select
}


function table_select_change(){
    table_name_value.value = table_select.value;
    create_check_box_list();
}


function save_table_change() {
    attribute_to_id_vis.style.display = "none";
    if (save_table_value.checked === true) {
        attribute_to_id_vis.style.display = "block"
    }
}


columns_list_value.style.display = "none";
table_select = init_table_name_value();
save_table_change();
init_check_all_box();
table_select_change();


let buttons = document.getElementsByClassName("qhana-form-buttons");
buttons[0].addEventListener("click", write_checkbox_content_to_str_list, false);
save_table_value.addEventListener("change", save_table_change);
table_select.addEventListener("change", table_select_change);