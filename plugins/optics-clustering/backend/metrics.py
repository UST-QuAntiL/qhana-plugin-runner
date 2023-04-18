from enum import Enum


class MetricEnum(Enum):
    braycurtis = "braycurtis"   # Ball_Tree, Brute
    canberra = "canberra"   # Ball_Tree, Brute
    chebyshev = "chebyshev" # Ball_Tree, KD_Tree, Brute
    cityblock = "cityblock" # Ball_Tree, KD_Tree, Brute
    correlation = "correlation" # Brute
    cosine = "cosine"   # Brute
    dice = "dice"   # Ball_Tree, Brute
    euclidean = "euclidean" # Ball_Tree, KD_Tree, Brute
    hamming = "hamming" # Ball_Tree, Brute
    haversine = "haversine" # Ball_Tree, Brute
    infinity = "infinity"   # Ball_Tree, KD_Tree
    jaccard = "jaccard" # Ball_Tree, Brute
    kulsinski = "kulsinski" # Ball_Tree, Brute
    l1 = "l1"   # Ball_Tree, KD_Tree, Brute
    l2 = "l2"   # Ball_Tree, KD_Tree, Brute
    mahalanobis = "mahalanobis" # Ball_Tree, Brute
    manhattan = "manhattan" # Ball_Tree, KD_Tree, Brute
    matching = "matching"   # Ball_Tree, Brute, Brute
    minkowski = "minkowski" # Ball_Tree, KD_Tree, Brute
    nan_euclidean = "nan_euclidean" # Brute
    p = "p" # Ball_Tree, KD_Tree
    # precomputed = "precomputed" # possible but not implemented
    rogerstanimoto = "rogerstanimoto"   # Ball_Tree, Brute
    russellrao = "russellrao"   # Ball_Tree, Brute
    seuclidean = "seuclidean"   # Ball_Tree, Brute
    sokalmichener = "sokalmichener" # Ball_Tree, Brute
    sokalsneath = "sokalsneath" # Ball_Tree, Brute
    sqeuclidean = "sqeuclidean" # Brute
    wminkowski = "wminkowski"   # Ball_Tree
    yule = "yule"   # Brute

    def get_metric(self) -> str:
        return str(self.value)
