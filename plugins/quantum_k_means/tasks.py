from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

from plugins.quantum_k_means import QKMeans
from plugins.quantum_k_means.backend.clustering import (
    Clustering,
    NegativeRotationQuantumKMeansClustering,
    DestructiveInterferenceQuantumKMeansClustering,
    StatePreparationQuantumKMeansClustering,
    PositiveCorrelationQuantumKMeansClustering,
)
from plugins.quantum_k_means.schemas import (
    InputParameters,
    InputParametersSchema,
    VariantEnum,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

import numpy as np

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{QKMeans.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new quantum k-means calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entity_points_url = input_params.entity_points_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entity_points_url='{entity_points_url}'"
    )
    clusters_cnt = input_params.clusters_cnt
    TASK_LOGGER.info(f"Loaded input parameters from db: clusters='{clusters_cnt}'")
    variant = input_params.variant
    TASK_LOGGER.info(f"Loaded input parameters from db: variant='{variant}'")

    # load data from file

    entity_points = open_url(entity_points_url).json()
    id_to_idx = {}

    idx = 0

    for ent in entity_points:
        if ent["ID"] in id_to_idx:
            raise ValueError("Duplicate ID: ", ent["ID"])

        id_to_idx[ent["ID"]] = idx
        idx += 1

    points_cnt = len(id_to_idx)
    dimensions = len(entity_points[0]["point"])
    points_arr = np.zeros((points_cnt, dimensions))

    for ent in entity_points:
        idx = id_to_idx[ent["ID"]]
        points_arr[idx] = ent["point"]

    algo: Clustering

    if variant == VariantEnum.negative_rotation:
        algo = NegativeRotationQuantumKMeansClustering(number_of_clusters=clusters_cnt)
    elif variant == VariantEnum.destructive_interference:
        algo = DestructiveInterferenceQuantumKMeansClustering(
            number_of_clusters=clusters_cnt
        )
    elif variant == VariantEnum.state_preparation:
        algo = StatePreparationQuantumKMeansClustering(number_of_clusters=clusters_cnt)
    elif variant == VariantEnum.positive_correlation:
        algo = PositiveCorrelationQuantumKMeansClustering(number_of_clusters=clusters_cnt)
    else:
        raise ValueError("Unknown variant.")

    clusters = algo.create_cluster(points_arr)

    entity_clusters = []

    for ent_id, idx in id_to_idx.items():
        entity_clusters.append({"ID": ent_id, "href": "", "cluster": int(clusters[idx])})

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entity_clusters, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "clusters.json",
            "clusters",
            "application/json",
        )

    return "Result stored in file"
