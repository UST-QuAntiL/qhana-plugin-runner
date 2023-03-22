import numpy as np
from sklearn.cluster import OPTICS

from celery.utils.log import get_task_logger


TASK_LOGGER = get_task_logger(__name__)


class OpticsClustering:
    """
    OPTICS Referenz : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

        OPTICS (Ordering Points To Identify the Clustering Structure), closely related to DBSCAN, finds
        core sample of high density and expands clusters from them [R2c55e37003fe-1]. Unlike DBSCAN, k-
        eeps cluster hierarchy for a variable neighborhood radius. Better suited for usage on large da-
        tasets than the current sklearn implementation of DBSCAN.
        Clusters are then extracted using a DBSCAN-like method (cluster_method = ‘dbscan’) or an automa-
        tic technique proposed in [R2c55e37003fe-1] (cluster_method = ‘xi’).
        This implementation deviates from the original OPTICS by first performing k-nearest-neighborhood
        searches on all points to identify core sizes, then computing only the distances to unprocessed
        points when constructing the cluster order. Note that we do not employ a heap to manage the exp-
        ansion candidates, so the time complexity will be O(n^2). Read more in the User Guide.

     Parameters:
        min_samples:        int > 1 or float between 0 and 1 (default=5)
                            The number of samples in a neighborhood for a point to be considered as a
                            core point. Also, up and down steep regions can’t have more then min_sam-
                            ples consecutive non-steep points. Expressed as an absolute number or a
                            fraction of the number of samples (rounded to be at least 2).

        max_eps:            float, optional (default=np.inf)
                            The maximum distance between two samples for one to be considered as in t-
                            he neighborhood of the other. Default value of np.inf will identify clust-
                            ers across all scales; reducing max_eps will result in shorter run times.

        metric:             str or callable, optional (default=’minkowski’)
                            Metric to use for distance computation. Any metric from scikit-learn or
                            scipy.spatial.distance can be used.

                            If metric is a callable function, it is called on each pair of instances
                            (rows) and the resulting value recorded. The callable should take two arr-
                            ays as input and return one value indicating the distance between them.
                            This works for Scipy’s metrics, but is less efficient than passing the me-
                            tric name as a string. If metric is “precomputed”, X is assumed to be a
                            distance matrix and must be square.

                            Valid values for metric are:

                            from scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’,
                            ‘manhattan’]

                            from scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’,
                            ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’,
                            ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’,
                            ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]

                            See the documentation for scipy.spatial.distance for details on these metrics.

        p:                  int, optional (default=2)
                            Parameter for the Minkowski metric from sklearn.metrics.pairwise_distances.
                            When p = 1, this is equivalent to using manhattan_distance (l1), and euclidea-
                            n_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

        metric_params:      dict, optional (default=None)
                            Additional keyword arguments for the metric function.

        cluster_method:     str, optional (default=’xi’)
                            The extraction method used to extract clusters using the calculated reachability
                            and ordering. Possible values are “xi” and “dbscan”.

        eps:                float, optional (default=None)
                            The maximum distance between two samples for one to be considered as in the neig-
                            hborhood of the other. By default it assumes the same value as max_eps. Used only
                            when cluster_method='dbscan'.

        xi:                 float, between 0 and 1, optional (default=0.05)
                            Determines the minimum steepness on the reachability plot that constitutes a clu-
                            ster boundary. For example, an upwards point in the reachability plot is defined
                            by the ratio from one point to its successor being at most 1-xi. Used only when
                            cluster_method='xi'.

     predecessor_correction:bool, optional (default=True)
                            Correct clusters according to the predecessors calculated by OPTICS [R2c55e37003fe-2].
                            This parameter has minimal effect on most datasets. Used only when cluster_method='xi'.

        min_cluster_size:   int > 1 or float between 0 and 1 (default=None)
                            Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a
                            fraction of the number of samples (rounded to be at least 2). If None, the value of
                            min_samples is used instead. Used only when cluster_method='xi'.

        algorithm:          {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
                            Algorithm used to compute the nearest neighbors:

                            ‘ball_tree’ will use BallTree
                            ‘kd_tree’ will use KDTree
                            ‘brute’ will use a brute-force search.
                            ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed
                             to fit method. (default)

                            Note: fitting on sparse input will override the setting of this parameter, using brute
                            force.

        leaf_size:          int, optional (default=30)
                            Leaf size passed to BallTree or KDTree. This can affect the speed of the construction
                            and query, as well as the memory required to store the tree. The optimal value depends on
                            the nature of the problem.

        n_jobs:             int or None, optional (default=None)
                            The number of parallel jobs to run for neighbors search. None means 1 unless in a
                            joblib.parallel_backend context. -1 means using all processors. See Glossary for more det-
                            ails.
    """

    def __init__(
        self,
        min_samples: float = 5,  # float between 0 and 1 else int
        max_eps: float = np.inf,
        metric: str = 'minkowski',
        p: int = 2,  # only when the minkowski metric is choosen
        metric_params: dict = None,  # additional keywords for the metric function
        cluster_method: str = 'xi',
        eps: float = None,  # by default it assumes the same value as max_eps (Only used when cluster_method='dbscan')
        xi: float = 0.05,  # only between 0 and 1 (Only used when cluster_method='xi')
        predecessor_correction: bool = True,  # only used when cluster_method='xi'
        min_cluster_size: float = None,  # float between 0 and 1 else int
        algorithm: str = 'auto',
        leaf_size: int = 30,  # only for BallTree or KDTree
        n_jobs: int = None  # -1 mean using all processors
    ):
        super().__init__()
        if min_samples <= 1 and min_samples >= 0:
            self.__min_samples: float = min_samples
        elif min_samples > 1:
            self.__min_samples: int = round(min_samples)
        else:
            TASK_LOGGER.error("min_samples is smaller than 0.")
            raise Exception("min_samples is smaller than 0")
        self.__max_eps: float = max_eps
        self.__metric: str = metric
        self.__p: int = p
        self.__metric_params: dict = None
        self.__cluster_method: str = cluster_method
        self.__eps: float = eps
        if xi >= 0 and xi <= 1:
            self.__xi: float = xi
        else:
            TASK_LOGGER.warning("xi is not between 0 and 1. Default Value was set! xi = 0.05")
            self.__xi: float = 0.05
        self.__predecessor_correction: bool = predecessor_correction
        self.__algorithm: str = algorithm
        self.__leaf_size: int = leaf_size
        self.__n_jobs: int = n_jobs
        if min_cluster_size == None or (min_cluster_size >= 0 and min_cluster_size <= 1):
            self.__min_cluster_size: float = min_cluster_size
        else:
            TASK_LOGGER.warning("min is not between 0 and 1 or None. Default Value was set! min_cluster_size = None")
            self.__min_cluster_size: float = None

        try:
            self.__cluster_instance: OPTICS = self.__create_optics_cluster_instance()
        except Exception as error:
            TASK_LOGGER.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")

        # sklearn.cluster._optics.OPTICS

    def __create_optics_cluster_instance(self) -> OPTICS:
        if self.__min_samples < 0:
            TASK_LOGGER.error("min_samples is smaller than 0.")
            raise Exception("min_samples is smaller than 0")
        elif self.__min_samples > 1:
            self.__min_samples = round(self.__min_samples)

        if self.__cluster_method != "xi" and self.__cluster_method != "dbscan":
            TASK_LOGGER.error("Not valid cluster_method.")
            raise Exception("Not valid cluster_method.")

        if self.__min_cluster_size != None and self.__min_cluster_size < 0 and self.__min_cluster_size > 1:
            TASK_LOGGER.warning("min is not between 0 and 1 or None. Default Value was set! min_cluster_size = None")
            self.__min_cluster_size: float = None

        if self.__algorithm != "auto" and self.__algorithm != "ball_tree" and self.__algorithm != "kd_tree" and self.__algorithm != "brute":
            TASK_LOGGER.error("Not valid algorithm method.")
            raise Exception("Not valid algorithm method.")

        if self.__cluster_method == "xi":
            if self.__xi > 1 and self.__xi < 0:
                TASK_LOGGER.warning("xi is not between 0 and 1. Default Value was set! xi = 0.05")
                self.__xi: float = 0.05

            if self.__algorithm == "ball_tree" or self.__algorithm == "kd_tree":

                if self.__metric == "minkowski":
                    # xi, ball algorithm , minkowski
                    return OPTICS(min_samples=self.__min_samples,
                                  max_eps=self.__max_eps,
                                  metric=self.__metric,
                                  p=self.__p,
                                  metric_params=self.__metric_params,
                                  cluster_method=self.__cluster_method,
                                  xi=self.__xi,
                                  predecessor_correction=self.__predecessor_correction,
                                  min_cluster_size=self.__min_cluster_size,
                                  algorithm=self.__algorithm,
                                  leaf_size=self.__leaf_size,
                                  n_jobs=self.__n_jobs
                                  )
                else:
                    # xi, ball algorithm , not minkowski
                    return OPTICS(min_samples=self.__min_samples,
                                  max_eps=self.__max_eps,
                                  metric=self.__metric,
                                  metric_params=self.__metric_params,
                                  cluster_method=self.__cluster_method,
                                  xi=self.__xi,
                                  predecessor_correction=self.__predecessor_correction,
                                  min_cluster_size=self.__min_cluster_size,
                                  algorithm=self.__algorithm,
                                  leaf_size=self.__leaf_size,
                                  n_jobs=self.__n_jobs
                                  )
            else:
                if self.__metric == "minkowski":
                    # xi, not ball algorithm, minkowski
                    return OPTICS(min_samples=self.__min_samples,
                                  max_eps=self.__max_eps,
                                  metric=self.__metric,
                                  p=self.__p,
                                  metric_params=self.__metric_params,
                                  cluster_method=self.__cluster_method,
                                  xi=self.__xi,
                                  predecessor_correction=self.__predecessor_correction,
                                  min_cluster_size=self.__min_cluster_size,
                                  algorithm=self.__algorithm,
                                  n_jobs=self.__n_jobs
                                  )
                else:
                    # xi, not ball algorithm , not minkowski
                    return OPTICS(min_samples=self.__min_samples,
                                  max_eps=self.__max_eps,
                                  metric=self.__metric,
                                  metric_params=self.__metric_params,
                                  cluster_method=self.__cluster_method,
                                  xi=self.__xi,
                                  predecessor_correction=self.__predecessor_correction,
                                  min_cluster_size=self.__min_cluster_size,
                                  algorithm=self.__algorithm,
                                  n_jobs=self.__n_jobs
                                  )


        elif self.__cluster_method == "dbscan":
            if self.__algorithm == "ball_tree" or self.__algorithm == "ball_tree":

                if self.__metric == "minkowski":
                    # dbscan, ball algorithm , minkowski
                    return OPTICS(min_samples=self.__min_samples,
                                  max_eps=self.__max_eps,
                                  metric=self.__metric,
                                  p=self.__p,
                                  metric_params=self.__metric_params,
                                  cluster_method=self.__cluster_method,
                                  eps=self.__eps,
                                  min_cluster_size=self.__min_cluster_size,
                                  algorithm=self.__algorithm,
                                  leaf_size=self.__leaf_size,
                                  n_jobs=self.__n_jobs
                                  )
                else:
                    # dbscan, ball algorithm , not minkowski
                    return OPTICS(min_samples=self.__min_samples,
                                  max_eps=self.__max_eps,
                                  metric=self.__metric,
                                  metric_params=self.__metric_params,
                                  cluster_method=self.__cluster_method,
                                  eps=self.__eps,
                                  min_cluster_size=self.__min_cluster_size,
                                  algorithm=self.__algorithm,
                                  leaf_size=self.__leaf_size,
                                  n_jobs=self.__n_jobs
                                  )

            else:
                if self.__metric == "minkowski":
                    # dbscan, not ball algorithm, minkowski
                    return OPTICS(min_samples=self.__min_samples,
                                  max_eps=self.__max_eps,
                                  metric=self.__metric,
                                  p=self.__p,
                                  metric_params=self.__metric_params,
                                  cluster_method=self.__cluster_method,
                                  eps=self.__eps,
                                  min_cluster_size=self.__min_cluster_size,
                                  algorithm=self.__algorithm,
                                  n_jobs=self.__n_jobs
                                  )
                else:
                    # dbscan, not ball algorithm , not minkowski
                    return OPTICS(min_samples=self.__min_samples,
                                  max_eps=self.__max_eps,
                                  metric=self.__metric,
                                  metric_params=self.__metric_params,
                                  cluster_method=self.__cluster_method,
                                  eps=self.__eps,
                                  min_cluster_size=self.__min_cluster_size,
                                  algorithm=self.__algorithm,
                                  n_jobs=self.__n_jobs
                                  )

    def create_cluster(self, position_matrix: np.matrix, similarity_matrix: np.matrix) -> np.matrix:
        try:
            self.__cluster_instance = self.__create_optics_cluster_instance()
        except Exception as error:
            TASK_LOGGER.error("An Exception occurs by OPTICS initialization: " + str(error))
            raise Exception("Exception occurs in Method __create_optics_instance by OPTICS initialization.")

        try:
            self.__cluster_instance.fit(position_matrix)
            return self.__cluster_instance.labels_
        except Exception as error:
            TASK_LOGGER.error("An Exception occurs by clustering the postion_matrix: " + str(error))
            raise Exception("Exception occurs in Method create_cluster by clustering the positon_matrix.")
