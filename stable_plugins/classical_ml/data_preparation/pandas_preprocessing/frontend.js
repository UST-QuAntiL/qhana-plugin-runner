
const preprocessing_steps_value = document.getElementById("preprocessing_steps");
preprocessing_steps_value.style.display = "none";
const preprocessing_steps_parent = preprocessing_steps_value.parentNode;
const preprocessing_steps_list = document.createElement("div");
preprocessing_steps_list.style.cssText = "overflow: auto; max-width: 25em; margin: 0em; padding-top: 0.3em; padding-bottom: 0.3em; flex-grow: 1;";
preprocessing_steps_list.setAttribute("id", "preprocessing_steps_list");
preprocessing_steps_parent.appendChild(preprocessing_steps_list);
let add_step_element;
// preprocessing_steps_parent.appendChild(add_step_element);

let options_and_parameters = {
    "drop na":
        [
            {"type": "number", "name": "axis", "label": "Axis", "description": "Determine if rows or columns which contain missing values are removed.\n- 0 drops rows\n- 1 drops columns.", "min": "0", "max": "1"}, // axis
            {"type": "number", "name": "threshold", "label": "Threshold", "description": "Requires that many non-NA values. Cannot be combined with how. If left empty, then all values may not be NA.", "min": "0"}, // threshold
            {"type": "string", "name": "subset","label": "Subset", "description": "Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include."}, //subset
        ],
    "fill na":
        [
            {"type": "string", "name": "fill_value", "label": "Value", "description": "The value to fill every na with."},
        ],
    "drop duplicates":
        [
            {"type": "string", "name": "subset", "label": "Subset", "description": "Only consider certain columns for identifying duplicates, by default use all of the columns."},
            {"type": "select", "name": "keep", "label": "Keep", "options": ["first", "last", "all"], "description": "Determines which duplicates (if any) to keep.\n- first: Drop duplicates except for the first occurrence.\n- last: Drop duplicates except for the last occurrence.\n- all : Drop all duplicates."},
        ]
}


function get_value_from_dict(dict, key, default_value) {
    if (dict.hasOwnProperty(key)) {
        let v = dict[key];
        if (v !== null && v !== undefined) {
            return v;
        }
    }
    return default_value
}


function get_parameter(param_dict, value= "") {
    console.log("get_parameter: ");
    console.log(value);
    let parameter = undefined;
    if (param_dict["type"] === "string") {
        parameter = document.createElement("input");
        parameter.setAttribute("type", "text");
    }
    else if (param_dict["type"] === "number") {
        parameter = document.createElement("input");
        parameter.setAttribute("type", "number");
        parameter.setAttribute("step", get_value_from_dict(param_dict, "step", undefined));
        parameter.setAttribute("min", get_value_from_dict(param_dict, "min", undefined));
        parameter.setAttribute("max", get_value_from_dict(param_dict, "max", undefined));
    }
    else if (param_dict["type"] === "select") {
        parameter = document.createElement("select");
        param_dict["options"].forEach(function(el) {
            let option = document.createElement("option")
            option.setAttribute("value", el);
            option.innerText = el.toString();
            if (el === value) {
                option.setAttribute("selected", "selected");
            }
            parameter.appendChild(option);
        });
    }

    let form_field = undefined;
    if (parameter !== undefined) {
        parameter.setAttribute("class", "qhana-form-input");
        parameter.setAttribute("name", param_dict["name"]);
        parameter.setAttribute("value", value);

        form_field = document.createElement("div");
        form_field.setAttribute("class", "qhana-form-field");

        let label = document.createElement("label");
        label.setAttribute("class", "qhana-form-label");
        label.innerText = param_dict["label"];

        let wrapper = document.createElement("div");
        wrapper.setAttribute("class", "qhana-input-wrapper");
        wrapper.appendChild(parameter);

        form_field.appendChild(label);
        form_field.appendChild(wrapper);

        if (param_dict["description"] !== undefined) {
            let description = document.createElement("div");
            description.setAttribute("class", "qhana-input-description");
            let text = document.createElement("p");
            text.innerText = param_dict["description"];
            description.appendChild(text);

            form_field.appendChild(description);
        }
    }
    return form_field
}


function get_input_parameters(option, value_dict= {}){
    console.log(option);
    let parameters = [];
    let parameter_dicts = get_value_from_dict(options_and_parameters, option, []);

    parameter_dicts.forEach(function (param_dict){
        let param = get_parameter(param_dict, get_value_from_dict(value_dict, param_dict["name"], ""));  // , value_dict[param["name"]]);
        if (param !== undefined) {
            parameters.push(param);
        }
    });

    return parameters;
}


function set_input_parameters(selectable, param_dict={}) {
    console.log("set_input_parameters");
    console.log(selectable);
    let parameters = get_input_parameters(selectable.value, param_dict);
    let input_parameter_list = selectable.parentNode.parentNode.children[1];
    for (let i = input_parameter_list.children.length - 1; i >= 0; i--){
        input_parameter_list.children[i].remove();
    }
    parameters.forEach(function(param) {
        input_parameter_list.append(param);
    });
    console.log(parameters);
}


function get_selectable(select_options, id=null){
    let selectable = document.createElement("select");
    if (id !== null) {
        selectable.setAttribute("id", id);
    }
    selectable.setAttribute("class", "qhana-form-input");
    // selectable.setAttribute("size", "3");

    // selectable.setAttribute("onfocus", "this.size=3;");
    // selectable.setAttribute("onblur", "this.size=1;");
    // selectable.setAttribute("onchange", "this.size=1; this.blur();");

    selectable.addEventListener("change", function (event) {
        set_input_parameters(event.target);
    });
    // selectable.setAttribute("select", "this.size=1; this.blur();")
    // selectable.setAttribute("onselect", "this.size=1; this.blur();")


    select_options.forEach(function(el) {
        let option = document.createElement("option")
        option.setAttribute("value", el);
        option.innerText = el.toString();
        selectable.appendChild(option);
    });

    let div_wrap = document.createElement("li");
    div_wrap.appendChild(selectable);

    return div_wrap;
}

function collapse_toggle(event){
    console.log("collapse toggle!");
    // collapsable.toggle("active");
    console.log(event.target.parentNode.children);
    let children = event.target.parentNode.children;
    for (let i = children.length - 1; i >= 0; i--) {
        let child = children[i];
        if (child.name === "values") {
            console.log("make invisible " + child.toString());
            if (child.style.display === "block") {
                child.style.display = "none";
            } else {
                child.style.display = "block";
            }
            break;
        }
    }
}

function make_collapsable(element) {
    element.addEventListener("click", collapse_toggle);
}


// Processing step. Children: [preprocessing_step=drop down menu, values=list with input values]
function add_preprocessing_step(param_dict={}){
    console.log("add_preprocessing_step: ");
    console.log(param_dict);
    let preprocessing_step = document.createElement("div");
    /*preprocessing_step.className = "qhana-form-input";*/
    preprocessing_step.className = "ul";
    // border: 1px solid #8f8f9d;
    preprocessing_step.style.cssText = "max-width: 25em; list-style-type: none; margin: 0em; padding: 0.0em; flex-grow: 1; overflow-x: hidden;";

    // let collaps_button = document.createElement("div");
    // collaps_button.style.cssText = "float: left; width: 2vw; height: 2vw; max-width: 15px; max-height: 15px; background-color: #8f8f9d; border: 1px solid #8f8f9d; margin: 0em; padding: 0.0em; flex-grow: 1;";
    // // collaps_button.innerText = "collapse";
    // collaps_button.addEventListener("click", collapse_toggle);

    let preprocessing_type = get_selectable(Object.keys(options_and_parameters));
    if (param_dict["option_type"] !== undefined) {
        preprocessing_type.children[0].value = param_dict["option_type"];
    }
    preprocessing_step.appendChild(preprocessing_type);

    let values = document.createElement("li");
    values.setAttribute("name", "values");
    values.style.cssText = "display: block; padding-left: 2em; padding-top: 0.3em; padding-bottom: 0.3em;";
    // values.style.display = "block";
    preprocessing_step.appendChild(values);
    // preprocessing_steps_list.appendChild(preprocessing_step);

    // make_collapsable(preprocessing_step);

    preprocessing_steps_list.appendChild(preprocessing_step);

    set_input_parameters(preprocessing_type.children[0], param_dict);
}


function compile_preprocessing_list() {
    let children = preprocessing_steps_list.children
    let compiled_list = [];
    for (let i = 0; i < children.length; i++) {
        let selected_option = children[i].children[0].children[0].value;
        console.log(children[i].children[0].children[0]);
        console.log(selected_option);
        let input_parameters_html = children[i].children[1].children;
        let input_parameters = {"option_type": selected_option};
        for (let j = 0; j < input_parameters_html.length; j++) {
            let input = input_parameters_html[j].children[1].children[0];
            let v = input.value;
            if (input.type === "number") {
                v = parseFloat(v);
            }
            input_parameters[input.name] = v;
        }

        compiled_list.push(input_parameters);
    }

    compiled_list = JSON.stringify(compiled_list);

    preprocessing_steps_value.value = compiled_list;
}


function init_preprocessing_list() {
    console.log("start init_preprocessing_list");
    console.log(preprocessing_steps_value.value);
    if (preprocessing_steps_value.value !== "") {
        let preprocessing_steps = JSON.parse(preprocessing_steps_value.value);
        console.log(preprocessing_steps);
        if (preprocessing_steps.length === 0) {
            add_preprocessing_step();
        }
        for (let i = 0; i < preprocessing_steps.length; i++) {
            console.log(preprocessing_steps[i]);
            add_preprocessing_step(preprocessing_steps[i]);
        }
    }
    else {
        add_preprocessing_step();
    }
    console.log("stop init_preprocessing_list");
}


init_preprocessing_list();

// compile_preprocessing_list();

let buttons = document.getElementsByClassName("qhana-form-buttons");
buttons[0].addEventListener("click", compile_preprocessing_list, false);
