# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import lru_cache
from io import BytesIO
from typing import Optional

import numpy as np
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_array,
    load_entities,
)
from qhana_plugin_runner.requests import get_mimetype, open_url

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


@CELERY.task(name=f"{RidgeLoss.instance.identifier}.load_data", bind=True)
def load_data(self, db_id: int):
    """Load the features and target arrays into DB for fast and efficient access."""
    TASK_LOGGER.info(f"Load data for optimization '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(task_data.data, dict)

    features_data_url = task_data.data["features_url"]
    target_data_url = task_data.data["target_url"]

    assert isinstance(features_data_url, str)
    assert isinstance(target_data_url, str)

    with open_url(features_data_url, stream=True) as x:
        mimetype = get_mimetype(x)
        if not mimetype:
            raise ValueError("Could not determine mimetype of x!")

        data = tuple(
            e.values
            for e in ensure_array(load_entities(x, mimetype=mimetype), strict=True)
        )
        x_array = np.array(data)
        key_x = f"{db_id}.features"
        x_dump = BytesIO()
        np.save(x_dump, x_array, allow_pickle=False)
        DataBlob.set_value(RidgeLoss.instance.name, key_x, x_dump.getvalue())
        task_data.data["features_key"] = key_x
        task_data.data["weights"] = x_array.shape[1]
        del data  # clear large data from memory faster
        del x_array
        del x_dump

    with open_url(target_data_url, stream=True) as y:
        mimetype = get_mimetype(y)
        if not mimetype:
            raise ValueError("Could not determine mimetype of y!")

        data = tuple(
            e.values[0]
            for e in ensure_array(load_entities(y, mimetype=mimetype), strict=True)
        )
        y_array = np.array(data)
        key_y = f"{db_id}.target"
        y_dump = BytesIO()
        np.save(y_dump, y_array, allow_pickle=False)
        DataBlob.set_value(RidgeLoss.instance.name, key_y, y_dump.getvalue())
        task_data.data["target_key"] = key_y
        del data  # clear large data from memory faster
        del y_array
        del y_dump

    task_data.clear_previous_step()

    task_data.save(commit=True)


@CELERY.task(name=f"{RidgeLoss.instance.identifier}.clear_task_data")
def clear_task_data(db_id: int):
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        return "Task data not found"

    assert isinstance(task_data.data, dict)

    # clear the complete cache as the data is no longer required in memory
    # may slow down other jobs running at the same time
    load_data_from_db.cache_clear()

    for key in ("features_key", "target_key"):
        data_key = task_data.data[key]
        # clear blob data that is no longer needed
        DataBlob.delete_value(RidgeLoss.instance.name, data_key)
    DB.session.commit()

    return "completed objective function task"


@lru_cache(maxsize=8)  # values may be large so avoid excessive caching...
def load_data_from_db(key: str) -> np.ndarray:
    """Load a numpy array from the database given the database key.

    Results of this function are cached for maximum efficiency!

    Args:
        key (str): the key the array is stored under

    Returns:
        np.ndarray: the numpy array
    """
    data = BytesIO(DataBlob.get_value(RidgeLoss.instance.name, key))
    return np.load(data, allow_pickle=False)
