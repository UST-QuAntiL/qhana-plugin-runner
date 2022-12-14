frontend_js = """
// Long block of just gathering elements
// If a variable ends with _value, it's the element itself and the value can be retrieved or set
var qknn_type_value = document.getElementById("variant");
var exp_itr_value = document.getElementById("exp_itr");

// If a vairable ends with _vis, it's the parentNode's parentNode and we can set the visibility
var exp_itr_vis = exp_itr_value.parentNode.parentNode;

function set_default_values() {
    exp_itr_value.value = "1";
}


function hide_all() {
    exp_itr_vis.style.display = 'none';
}


function show_basheer_hamming() {
    exp_itr_vis.style.display = 'block';
}


function qknn_type_change(reset_values=true) {
    hide_all();
    if (reset_values) {
        set_default_values();
    }
    if (qknn_type_value.value === "basheer_hamming_qknn") {
        show_basheer_hamming();
    }
}

qknn_type_change(false);

qknn_type_value.onchange = qknn_type_change;
"""
