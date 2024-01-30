import urllib.parse

from celery.utils.log import get_task_logger
from requests import Response, post

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

_name = "qhana-plugin-runner-interactions"

TASK_LOGGER = get_task_logger(_name)


def make_callback(callback_url: str, callback_data) -> Response:
    """Make a callback to the given callback_url with the given callback_data."""
    callback_url = urllib.parse.unquote(callback_url)

    response = post(callback_url, json=callback_data)
    return response


@CELERY.task(name=f"{_name}.callback-task", bind=True, ignore_result=True)
def callback_task(self, task_log: str, callback_url: str, callback_data):
    """Make a callback to the given callback_url with the given callback_data."""
    make_callback(callback_url, callback_data)


@CELERY.task(name=f"{_name}.invoke-task", bind=True, ignore_result=True)
def invoke_task(
    self,
    task_log: str,
    step_id: str,
    db_id: int,
    href: str,
    ui_href: str,
    callback_url: str,
    prog_value: int,
    prog_start: int = 0,
    prog_target: int = 100,
    prog_unit: str = "%",
):
    """Invoke a plugin in a multistep plugin that calls another plugin.

    Attributes:
        step_id (str): ID of step, e.g., ``"step1"`` or ``"step1.step2b"``.
        href (str): The *absolute* URL of the REST entry point resource.
        ui_href (str): The *absolute* URL of the micro frontend that corresponds to the REST entry point resource.
        callback_url (str): The *absolute* URL of the callback endpoint of the invoking plugin.
        prog_value (int): progress value.
        prog_start (int): progress start value.
        prog_target (int): progress target value.
        prog_unit (str): progress unit(default: "%").
    """
    if not isinstance(task_log, str):
        raise TypeError(
            f"The task log / task metadata must be of type str to be stored in the database! (expected str but got {type(task_log)})"
        )

    TASK_LOGGER.debug(f"Adding next step with db id '{db_id}'")
    task_data: ProcessingTask = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        # TODO use better fitting error
        raise KeyError(f"Could not find db entry for id {db_id}, add_step failed!")

    task_data.progress_start = prog_start
    task_data.progress_target = prog_target
    task_data.progress_unit = prog_unit
    task_data.progress_value = prog_value

    href = urllib.parse.unquote(href)
    href = urllib.parse.urlparse(href)
    href = href._replace(query=f"callbackUrl={callback_url}&{href.query}")
    href = urllib.parse.urlunparse(href)

    ui_href = urllib.parse.unquote(ui_href)
    ui_href = urllib.parse.urlparse(ui_href)
    ui_href = ui_href._replace(query=f"callbackUrl={callback_url}&{ui_href.query}")
    ui_href = urllib.parse.urlunparse(ui_href)

    task_data.add_next_step(href=href, ui_href=ui_href, step_id=step_id)

    if isinstance(task_log, str):
        task_data.add_task_log_entry(task_log)
    else:
        task_data.add_task_log_entry(repr(task_log))

    task_data.save(commit=True)
    TASK_LOGGER.debug(f"Save task log for task with db id '{db_id}' successful.")
