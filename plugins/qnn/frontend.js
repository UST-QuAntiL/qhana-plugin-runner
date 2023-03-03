// use default dataset
var use_default_dataset = document.getElementById("use_default_dataset");
// custom dataset files
var train_points_url = document.getElementById("train_points_url").parentNode.parentNode;
var train_label_points_url = document.getElementById("train_label_points_url").parentNode.parentNode;
var test_points_url = document.getElementById("test_points_url").parentNode.parentNode;
var test_label_points_url = document.getElementById("test_label_points_url").parentNode.parentNode;

function use_default_dataset_change() {
    if (use_default_dataset.checked === true) {
        // no data files required
        train_points_url.style.display = 'none';
        train_label_points_url.style.display = 'none';
        test_points_url.style.display = 'none';
        test_label_points_url.style.display = 'none';
    } else {
        train_points_url.style.display = 'block';
        train_label_points_url.style.display = 'block';
        test_points_url.style.display = 'block';
        test_label_points_url.style.display = 'block';
    }
}

// visualize result?
var visualize_classification = document.getElementById("visualize");
// resolution of the visualization
var resolution = document.getElementById("resolution").parentNode.parentNode;
function visualize_classification_change() {
    if (visualize_classification.checked === true) {
        resolution.style.display = 'block';
    } else {
        resolution.style.display = 'none';
    }
}



// use quantum or classical NN
var network_enum = document.getElementById("network_enum");
// quantum parameters
var backend = document.getElementById("device").parentNode.parentNode;
var ibmq_token = document.getElementById("ibmq_token").parentNode.parentNode;
var custom_backend = document.getElementById("custom_backend").parentNode.parentNode;
var shots = document.getElementById("shots").parentNode.parentNode;



// if backendname starts with "aer" ->hide ibmqtoken and custombackendname
// if backendname starts with "ibmq" -> hide custombackendname
// if backendname starts with "custom_ibmq" -> show all
backend_type = document.getElementById("device");
function backend_type_change() {
    backend.style.display = 'none';
    ibmq_token.style.display = 'none';
    custom_backend.style.display = 'none';
    shots.style.display = 'none';
    if (network_enum.value === "dressed_quantum_net") {
        backend.style.display = 'block';
        shots.style.display = 'block';
        if (backend_type.value.startsWith("ibmq")) {
            ibmq_token.style.display = 'block';
            custom_backend.style.display = 'none';
        } else if (!backend_type.value.startsWith("aer")) {
            ibmq_token.style.display = 'block';
            custom_backend.style.display = 'block';
        }
    }
}

var n_qubits = document.getElementById("n_qubits").parentNode.parentNode;
var q_depth = document.getElementById("q_depth").parentNode.parentNode;
var preprocess_layers = document.getElementById("preprocess_layers").parentNode.parentNode;
var postprocess_layers = document.getElementById("postprocess_layers").parentNode.parentNode;
var hidden_layers = document.getElementById("hidden_layers").parentNode.parentNode;
var q_weights_to_wiggle = document.getElementById("weights_to_wiggle").parentNode.parentNode;
var diff_method = document.getElementById("diff_method").parentNode.parentNode;
function use_quantum_change() {
    backend_type_change()
    n_qubits.style.display = 'none';
    q_depth.style.display = 'none';
    preprocess_layers.style.display = 'none';
    postprocess_layers.style.display = 'none';
    hidden_layers.style.display = 'none';
    q_weights_to_wiggle.style.display = 'none';
    diff_method.style.display = 'none';
    if (network_enum.value === "dressed_quantum_net") {
        // show quantum elements
        n_qubits.style.display = 'block';
        q_depth.style.display = 'block';
        preprocess_layers.style.display = 'block';
        postprocess_layers.style.display = 'block';
        q_weights_to_wiggle.style.display = 'block';
        diff_method.style.display = 'block';
    } else if (network_enum.value === "feed_forward_net") {
        // hide unnecessary elements
        hidden_layers.style.display = 'block';
    }
}

use_quantum_change();
use_default_dataset_change();
visualize_classification_change();
backend_type_change();

// react to changes
network_enum.addEventListener("change", use_quantum_change);
use_default_dataset.addEventListener("change", use_default_dataset_change);
visualize_classification.addEventListener("change", visualize_classification_change);
backend_type.addEventListener("change", backend_type_change);
