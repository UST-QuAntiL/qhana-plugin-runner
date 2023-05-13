import mimetypes
from tempfile import SpooledTemporaryFile
from typing import Iterator, Optional

import numpy as np
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from scipy.optimize import minimize

from . import RidgeLoss

TASK_LOGGER = get_task_logger(__name__)


def ridge_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """
    Calculate the ridge loss given weights, features, target, and alpha.

    Args:
        w: Weights
        X: Features
        y: Target
        alpha: Ridge regularization parameter

    Returns:
        The calculated ridge loss.
    """
    y_pred = np.dot(X, w)
    mse = np.mean((y - y_pred) ** 2)
    ridge_penalty = alpha * np.sum(w**2)
    return mse + ridge_penalty


def minimize_(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Optimize the input data using ridge loss as the objective function.

    Args:
        X: Features
        y: Target
        alpha: Ridge regularization parameter

    Returns:
        Optimized weights.
    """
    initial_weights = np.random.randn(X.shape[1])
    result = minimize(ridge_loss, initial_weights, args=(X, y, alpha), method="L-BFGS-B")
    return result.x


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
        file_type = file_.headers["Content-Type"]
        entities_generator = load_entities(file_, mimetype=file_type)
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


@CELERY.task(name=f"{RidgeLoss.instance.identifier}.optimize", bind=True)
def optimize(self, db_id: int) -> str:
    """
    Optimize the input data using the given hyperparameters, with ridge loss as the objective function.
    Args:
        self: The task instance.
        db_id: Database ID that will be used to retrieve the task data from the database.

    Returns:
        A log message.
    """
    TASK_LOGGER.info(f"Starting the optimization task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    alpha: float = task_data.data.get("alpha")
    input_file_url: str = task_data.data.get("input_file_url")
    target_variable_name: str = task_data.data.get("target_variable")

    X, y = get_features_and_target(input_file_url, target_variable_name)

    w = minimize_(X, y, alpha)

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entities=[{"w": w.tolist()}], file_=output, mimetype="text/csv", attributes=["w"])
        STORE.persist_task_result(
            db_id,
            output,
            "output_ridgeloss.csv",
            "objective-function-output",
            "text/csv",
        )
    return "Optimization finished."
