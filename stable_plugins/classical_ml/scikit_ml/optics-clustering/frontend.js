var algorithm_enum_val = document.getElementById("algorithm_enum");
var metric_enum_val = document.getElementById("metric_enum");
var method_enum_val = document.getElementById("method_enum");
var epsilon_val = document.getElementById("epsilon");
var xi_val = document.getElementById("xi");
var min_cluster_size_val = document.getElementById("min_cluster_size");
var leaf_size_val = document.getElementById("leaf_size");
var minkowski_p_val = document.getElementById("minkowski_p");


var epsilon_dis = epsilon_val.parentNode.parentNode;
var xi_dis = xi_val.parentNode.parentNode;
var predecessor_correction_dis = document.getElementById("predecessor_correction").parentNode.parentNode;
var min_cluster_size_dis = min_cluster_size_val.parentNode.parentNode;
var leaf_size_dis = leaf_size_val.parentNode.parentNode;
var minkowski_p_dis = minkowski_p_val.parentNode.parentNode;

function set_union(setA, setB) {
    const _union = new Set(setA);
    setB.forEach(elem => _union.add(elem));
    return _union;
}

//seuclidean, wminkowski, mahalanobis need more params
const ball_tree_metrics = new Set(["braycurtis", "canberra", "chebyshev", "cityblock", "dice", "euclidean", "hamming", "haversine", "jaccard", "kulsinski", "l1", "l2", "mahalanobis", "manhattan", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "wminkowski"]);
const kd_tree_metrics = new Set(['chebyshev', 'cityblock', 'euclidean', 'l1', 'l2', 'manhattan', 'minkowski']);
const brute_metrics = new Set(['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'haversine', 'jaccard', 'kulsinski', 'l1', 'l2', 'mahalanobis', 'manhattan', 'matching', 'minkowski', 'nan_euclidean', 'precomputed', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']);
const auto_metrics = set_union(set_union(ball_tree_metrics, kd_tree_metrics), brute_metrics);


function set_metric_options(active_options) {
    metric_enum_val.childNodes.forEach(metric => {
        metric['disabled'] = !active_options.has(metric.value);
    });
    metric_enum_val.value = active_options.values().next().value;
}


function algorithm_enum_change() {
    leaf_size_dis.style.display = "block"
    if (algorithm_enum_val.value === "ball_tree") {
        set_metric_options(ball_tree_metrics);
    }
    else if (algorithm_enum_val.value === "kd_tree") {
        set_metric_options(kd_tree_metrics);
    }
    else if (algorithm_enum_val.value === "brute") {
        leaf_size_dis.style.display = "none"
        leaf_size_val.value = 30;
        set_metric_options(brute_metrics);
    }
    else if (algorithm_enum_val.value === "auto") {
        set_metric_options(auto_metrics);
    }
}


function method_enum_change() {
    epsilon_dis.style.display = "none";
    xi_dis.style.display = "none";
    predecessor_correction_dis.style.display = "none";
    min_cluster_size_dis.style.display = "none";
    if (method_enum_val.value === "dbscan") {
        epsilon_dis.style.display = "block";

        xi_val.value = 0.05;
        min_cluster_size_val.value = -1;
    }
    else if (method_enum_val.value === "xi") {
        xi_dis.style.display = "block";
        predecessor_correction_dis.style.display = "block";
        min_cluster_size_dis.style.display = "block";

        epsilon_val.value = -1;
    }
}


function metric_enum_change() {
    minkowski_p_dis.style.display = "none";
    if (metric_enum_val.value === "minkowski") {
        minkowski_p_dis.style.display = "block";
    }
    else {
        minkowski_p_val.value = 2;
    }
}


algorithm_enum_val.addEventListener("change", algorithm_enum_change);
algorithm_enum_change();
method_enum_val.addEventListener("change", method_enum_change);
method_enum_change();
metric_enum_val.addEventListener("change", metric_enum_change);
metric_enum_change();