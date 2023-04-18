var algorithm_enum_val = document.getElementById("algorithm_enum");
var metric_enum_val = document.getElementById("metric_enum");

function set_union(setA, setB) {
    const _union = new Set(setA);
    setB.forEach(elem => _union.add(elem));
    return _union;
}

const ball_tree_metrics = new Set(["braycurtis", "canberra", "chebyshev", "cityblock", "dice", "euclidean", "hamming", "haversine", "infinity", "jaccard", "kulsinski", "l1", "l2", "mahalanobis", "manhattan", "matching", "minkowski", "p", "pyfunc", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "wminkowski"]);
const kd_tree_metrics = new Set(['chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p']);
const brute_metrics = new Set(['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'haversine', 'jaccard', 'kulsinski', 'l1', 'l2', 'mahalanobis', 'manhattan', 'matching', 'minkowski', 'nan_euclidean', 'precomputed', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']);
const auto_metrics = set_union(set_union(ball_tree_metrics, kd_tree_metrics), brute_metrics);


function set_metric_options(active_options) {
    metric_enum_val.childNodes.forEach(metric => {
        metric['disabled'] = !active_options.has(metric.value);
    });
    metric_enum_val.value = active_options.values().next().value;
}


function algorithm_enum_change() {
    console.log("change algorithm enum")
    if (algorithm_enum_val.value === "ball_tree") {
        set_metric_options(ball_tree_metrics);
    }
    else if (algorithm_enum_val.value === "kd_tree") {
        set_metric_options(kd_tree_metrics);
    }
    else if (algorithm_enum_val.value === "brute") {
        set_metric_options(brute_metrics);
    }
    else if (algorithm_enum_val.value === "auto") {
        set_metric_options(auto_metrics);
    }
}


algorithm_enum_val.addEventListener("change", algorithm_enum_change);
algorithm_enum_change();
