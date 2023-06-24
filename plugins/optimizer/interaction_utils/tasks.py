import urllib.parse
from requests import Response, post

from qhana_plugin_runner.celery import CELERY

_name = "qhana-plugin-runner-interactions"


def make_callback(callback_url: str, callback_data) -> Response:
    """Make a callback to the given callback_url with the given callback_data."""
    callback_url = urllib.parse.unquote(callback_url)

    response = post(callback_url, json=callback_data)
    return response


@CELERY.task(name=f"{_name}.callback-task", bind=True, ignore_result=True)
def callback_task(self, task_log: str, callback_url: str, callback_data):
    """Make a callback to the given callback_url with the given callback_data."""
    make_callback(callback_url, callback_data)
