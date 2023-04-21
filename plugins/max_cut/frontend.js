var max_cut_enum_val = document.getElementById("max_cut_enum");

var optimizer_dis = document.getElementById("optimizer").parentNode.parentNode;
var max_trials_dis = document.getElementById("max_trials").parentNode.parentNode;
var reps_dis = document.getElementById("reps").parentNode.parentNode;
var entanglement_pattern_enum_dis = document.getElementById("entanglement_pattern_enum").parentNode.parentNode;
var backend_dis = document.getElementById("backend").parentNode.parentNode;
var shots_dis = document.getElementById("shots").parentNode.parentNode;
var ibmq_custom_backend_dis = document.getElementById("ibmq_custom_backend").parentNode.parentNode;
var ibmq_token_dis = document.getElementById("ibmq_token").parentNode.parentNode;

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
        ibmq_custom_backend_dis.style.display = "block";
        ibmq_token_dis.style.display = "block";
    }
}

max_cut_enum_change();
max_cut_enum_val.addEventListener("change", max_cut_enum_change);
