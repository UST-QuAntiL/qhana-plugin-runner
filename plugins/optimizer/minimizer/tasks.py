from typing import Iterator, Optional

from celery.utils.log import get_task_logger
import numpy as np
import requests
from scipy.optimize import minimize as scipy_minimize
from ..shared.schemas import (
    CalcInputData,
    CalcInputDataSchema,
    LossResponseData,
    LossResponseSchema,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import ensure_dict, load_entities

from qhana_plugin_runner.requests import open_url

from . import Minimizer
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask


TASK_LOGGER = get_task_logger(__name__)


def get_features(ent: dict, target_variable_name: str) -> np.ndarray:
    """
    Get the feature vector for a given entity.

    Args:
        ent: A dictionary representing an entity.
        target_variable_name: The name of the target variable.

    Returns:
        A numpy array representing the feature vector for the given entity.
    """
    dimension_keys = [
        k for k in ent.keys() if k not in ("ID", "href", target_variable_name)
    ]
    dimension_keys.sort()
    point = np.empty(len(dimension_keys))
    for idx, d in enumerate(dimension_keys):
        point[idx] = ent[d]
    return point


def get_entity_generator(
    entity_points_url: str, target_variable_name: str
) -> Iterator[dict]:
    """
    Return a generator for the entity points, given a URL to them.

    Args:
        entity_points_url: URL to the entity points
        target_variable_name: Name of the target variable

    Yields:
        A dictionary representing an entity.
    """
    with open_url(entity_points_url) as file_:
        file_.encoding = "utf-8"
        file_.headers["Content-Type"] = "text/csv"
        entities_generator = load_entities(file_, mimetype="text/csv")
        entities_generator = ensure_dict(entities_generator)
        for ent in entities_generator:
            yield {
                "ID": ent["ID"],
                "href": ent.get("href", ""),
                "features": get_features(ent, target_variable_name),
                "target": float(ent[target_variable_name]),
            }


def get_features_and_target(
    entity_points_url: str, target_variable_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return NumPy arrays for the entity points' features and target variable, given a URL to them.

    Args:
        entity_points_url: URL to the entity points
        target_variable_name: Name of the target variable

    Returns:
        A tuple containing NumPy arrays for the features and target variables
    """
    x = []
    y = []
    for ent in get_entity_generator(entity_points_url, target_variable_name):
        x.append(ent["features"])
        y.append(ent["target"])
    return np.array(x), np.array(y)


def loss_(loss_calc_endpoint_url):
    def loss(x, y, x0, hyperparameters):
        request_schema = CalcInputDataSchema()
        request_data = request_schema.dump(
            CalcInputData(x0=x0, x=x, y=y, hyperparameters=hyperparameters)
        )

        response = requests.post(loss_calc_endpoint_url, json=request_data)
        response_schema = LossResponseSchema()
        response_data: LossResponseData = response_schema.load(response.json())
        return response_data.loss

    return loss


@CELERY.task(name=f"{Minimizer.instance.identifier}.minimize", bind=True)
def minimize_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting the optimization task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_file_url: str = task_data.data.get("input_file_url")
    target_variable_name: str = task_data.data.get("target_variable")
    loss_calc_endpoint_url = task_data.data.get("calc_loss_endpoint_url")
    hyperparameters = task_data.data.get("hyperparameters")

    X, y = get_features_and_target(input_file_url, target_variable_name)

    loss_fun = loss_(loss_calc_endpoint_url)

    initial_weights = np.random.randn(X.shape[1])
    result = scipy_minimize(
        loss_fun, initial_weights, args=(X, y, hyperparameters), method="L-BFGS-B"
    )

    TASK_LOGGER.info(f"Optimization result: {result}")
    return ", ".join(map(str, result.x.flatten()))
