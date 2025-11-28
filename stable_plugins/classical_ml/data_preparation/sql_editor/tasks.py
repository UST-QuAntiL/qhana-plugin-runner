from typing import Literal, Optional

from celery.utils.log import get_task_logger
from flask.globals import current_app
from requests import request

from qhana_plugin_runner.celery import CELERY

from . import plugin

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{plugin.SQLEditor.instance.identifier}.process_sql",
    bind=True,
    ignore_result=True,
)
def process_sql(self):
    pass
