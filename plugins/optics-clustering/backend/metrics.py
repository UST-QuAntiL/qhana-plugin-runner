from enum import Enum


class MetricEnum(Enum):
    braycurtis = "braycurtis"
    canberra = "canberra"
    cdist = "cdist"
    chebyshev = "chebyshev"
    cityblock = "cityblock"
    correlation = "correlation"
    cosine = "cosine"
    dice = "dice"
    directed_hausdorff = "directed_hausdorff"
    euclidean = "euclidean"
    hamming = "hamming"
    is_valid_dm = "is_valid_dm"
    is_valid_y = "is_valid_y"
    jaccard = "jaccard"
    jensenshannon = "jensenshannon"
    kulsinski = "kulsinski"
    kulczynski1 = "kulczynski1"
    mahalanobis = "mahalanobis"
    minkowski = "minkowski"
    num_obs_dm = "num_obs_dm"
    num_obs_y = "num_obs_y"
    pdist = "pdist"
    rogerstanimoto = "rogerstanimoto"
    russellrao = "russellrao"
    seuclidean = "seuclidean"
    sokalmichener = "sokalmichener"
    sokalsneath = "sokalsneath"
    sqeuclidean = "sqeuclidean"
    squareform = "squareform"
    yule = "yule"

    def get_metric(self) -> str:
        return str(self.value)
