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

from typing import Optional
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

from celery.utils.log import get_task_logger
from flask.globals import current_app

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.interop import (
    call_plugin_endpoint,
    get_task_result_no_wait,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import (
    TASK_STATUS_CHANGED,
    TASK_STEPS_CHANGED,
    add_step,
    save_task_error,
    save_task_result,
)

from . import Optimizer

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{Optimizer.instance.identifier}.add_plugin_entrypoint_task", bind=True
)
def add_plugin_entrypoint_task(
    self, db_id: int, plugin_url: str, webhook_url: str, step_id: str, task_log: str
):
    with open_url(plugin_url) as metadata:
        decoded = metadata.json()
        href = decoded.get("entryPoint").get("href")
        href = urljoin(plugin_url, href)
        ui_href = decoded.get("entryPoint").get("uiHref")
        ui_href = urljoin(plugin_url, ui_href)

    # generate callback query param
    callback_query = "callback=" + quote(webhook_url, safe="")

    # add callback to href
    *href_parts, href_query, _ = urlsplit(href)
    href_query = f"{href_query}&{callback_query}" if href_query else callback_query
    href = urlunsplit([*href_parts, href_query, ""])
    # add callback to ui href
    *ui_href_parts, ui_href_query, _ = urlsplit(ui_href)
    ui_href_query = (
        f"{ui_href_query}&{callback_query}" if ui_href_query else callback_query
    )
    ui_href = urlunsplit([*ui_href_parts, ui_href_query, ""])

    return self.replace(
        add_step.s(
            task_log=task_log, db_id=db_id, step_id=step_id, href=href, ui_href=ui_href
        )
    )


@CELERY.task(
    name=f"{Optimizer.instance.identifier}.check_of_steps", bind=True, ignore_result=True
)
def check_of_steps(self, db_id: int):
    TASK_LOGGER.info(f"Starting passing data to of plugin with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(task_data.data, dict)

    task_state = task_data.data["task_state"]

    if task_state not in ("of_setup", "of_cleanup"):
        # ignore step updates of objective function if not in the
        # objective function setup phase
        return

    of_task_url = task_data.data["of_task_url"]

    assert isinstance(of_task_url, str)

    status, of_data = get_task_result_no_wait(of_task_url)

    if status != "PENDING":
        return  # no need to handle step updates if result is already settled

    cleared_of_steps = set()
    uncleared_of_step = None

    # parse of steps
    for index, step in enumerate(of_data.get("steps", [])):
        external_step_id = step.get("stepId", str(index))
        step_id = f"of.{external_step_id}"
        if step.get("cleared"):
            if external_step_id in ("pass_data", "evaluate"):
                continue  # ignore the special steps
            cleared_of_steps.add(step_id)
        else:
            step["internal_step_id"] = step_id
            uncleared_of_step = step

    # mark cleared steps as cleared
    did_clear_step = False
    for step in task_data.steps:
        if step.step_id in cleared_of_steps and not step.cleared:
            did_clear_step = True
            step.cleared = True
            DB.session.add(step)

    # handle uncleared step (if normal step)
    if uncleared_of_step and (step_id := uncleared_of_step.get("stepId")) not in (
        "pass_data",
        "evaluate",
    ):
        task_data.add_next_step(
            href=urljoin(of_task_url, uncleared_of_step["href"]),
            ui_href=urljoin(of_task_url, uncleared_of_step["uiHref"]),
            step_id=uncleared_of_step["internal_step_id"],
        )
        uncleared_of_step = None

    if uncleared_of_step and uncleared_of_step.get("stepId") == "evaluate":
        # objective function is fully setup, enter minimizer init phase
        task_data.data["task_state"] = "minimizer_init"

    DB.session.commit()

    app = current_app._get_current_object()
    if did_clear_step:
        TASK_STEPS_CHANGED.send(app, task_id=db_id)

    # handle special steps
    if uncleared_of_step:
        step_id = uncleared_of_step.get("stepId")
        if step_id == "pass_data":
            call_plugin_endpoint(
                urljoin(of_task_url, uncleared_of_step["href"]),
                data={
                    "features": task_data.data["features_url"],
                    "target": task_data.data["target_url"],
                },
                debug=True,
            )
        if step_id == "evaluate":
            if task_data.has_uncleared_step:
                if task_data.steps[-1].step_id == "minimizer_setup":
                    # step is already present, do nothing
                    return
            task = add_plugin_entrypoint_task.s(
                db_id=db_id,
                plugin_url=task_data.data["minimizer_plugin_url"],
                webhook_url=task_data.data["minimizer_webhook"],
                step_id="minimizer_setup",
                task_log="Prepare to setup minimizer.",
            )

            task.link_error(save_task_error.s(db_id=db_id))

            return self.replace(task)


@CELERY.task(
    name=f"{Optimizer.instance.identifier}.check_minimizer_steps",
    bind=True,
    ignore_result=True,
)
def check_minimizer_steps(self, db_id: int):
    TASK_LOGGER.info(f"Starting passing data to minimizer plugin with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(task_data.data, dict)

    task_state = task_data.data["task_state"]

    if task_state not in ("minimizer_setup", "minimize"):
        # ignore step updates of minimizer if not in any minimizer phase
        return

    minimizer_task_url = task_data.data["minimizer_task_url"]

    assert isinstance(minimizer_task_url, str)

    status, minimizer_data = get_task_result_no_wait(minimizer_task_url)

    if status != "PENDING":
        return  # no need to handle step updates if result is already settled

    cleared_minimizer_steps = set()
    uncleared_minimizer_step = None

    # parse of steps
    for index, step in enumerate(minimizer_data.get("steps", [])):
        external_step_id = step.get("stepId", str(index))
        step_id = f"min.{external_step_id}"
        if step.get("cleared"):
            if external_step_id != "minimize":
                continue  # ignore the special steps
            cleared_minimizer_steps.add(step_id)
        else:
            step["internal_step_id"] = step_id
            uncleared_minimizer_step = step

    # mark cleared steps as cleared
    did_clear_step = False
    for step in task_data.steps:
        if step.step_id in cleared_minimizer_steps and not step.cleared:
            did_clear_step = True
            step.cleared = True
            DB.session.add(step)

    # handle uncleared step (if normal step)
    if (
        uncleared_minimizer_step
        and (step_id := uncleared_minimizer_step.get("stepId")) != "minimize"
    ):
        task_data.add_next_step(
            href=urljoin(minimizer_task_url, uncleared_minimizer_step["href"]),
            ui_href=urljoin(minimizer_task_url, uncleared_minimizer_step["uiHref"]),
            step_id=uncleared_minimizer_step["internal_step_id"],
        )
        uncleared_minimizer_step = None
        if step_id == "evaluate":
            # objective function is fully setup, enter minimizer setup phase
            task_data.data["task_state"] = "minimizer_setup"

    if uncleared_minimizer_step and uncleared_minimizer_step.get("stepId") == "minimize":
        # minimizer is ready to minimize, enter minimize phase
        task_data.data["task_state"] = "minimize"

    DB.session.commit()

    app = current_app._get_current_object()
    if did_clear_step:
        TASK_STEPS_CHANGED.send(app, task_id=db_id)

    # handle special steps
    if uncleared_minimizer_step:
        step_id = uncleared_minimizer_step.get("stepId")
        if step_id == "minimize":
            call_plugin_endpoint(
                urljoin(minimizer_task_url, uncleared_minimizer_step["href"]),
                data={
                    "objectiveFunction": task_data.data["of_task_url"],
                    # TODO: "initialWeights": "TODO"
                },
                debug=True,
            )


@CELERY.task(
    name=f"{Optimizer.instance.identifier}.enter_of_cleanup",
    bind=True,
    ignore_result=True,
)
def enter_of_cleanup(self, db_id: int):
    TASK_LOGGER.info(
        f"Start cleaning up the running objective function task for task with db_id={db_id}"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(task_data.data, dict)

    if task_data.data["task_state"] != "of_cleanup":
        return  # called cleanup in the wrong tasl phase

    if task_data.data.get("of_success"):
        return  # task is already finished, no cleanup required

    result_url = task_data.data["of_task_url"]

    assert isinstance(result_url, str)

    status, result_data = get_task_result_no_wait(result_url)

    if status != "PENDING":
        return  # task is already finished, no cleanup required

    steps = result_data.get("steps", [])

    if not steps:
        return  # task has no steps, no cleanup possible

    last_step = steps[-1]

    if last_step.get("cleared"):
        return  # task has no uncleared steps, no cleanup required

    if last_step.get("stepId") == "evaluate":
        # of is still in evaluate step, call step href to advance of progress
        # TODO: maybe pass final weights to evaluate step?
        call_plugin_endpoint(urljoin(result_url, last_step["href"]), None)

    else:
        # of is in any other open step, make sure that step is checked
        return self.replace(check_of_steps.s(db_id=db_id))


@CELERY.task(
    name=f"{Optimizer.instance.identifier}.handle_minimizer_result",
    bind=True,
    ignore_result=True,
)
def handle_minimizer_result(self, db_id: int):
    TASK_LOGGER.info(f"Start gathering minimizer result for task with db_id={db_id}")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(task_data.data, dict)

    result_url = task_data.data["minimizer_task_url"]

    status, result_data = get_task_result_no_wait(result_url)

    if status == "PENDING":
        return  # result was in fact not settled

    if status == "FAILURE":
        task_data.data["minimizer_success"] = False
        task_data.save(commit=True)
        return self.replace(check_final_result.s(db_id=db_id))

    outputs = result_data.get("outputs", [])

    if not task_data.data.get("minimizer_success"):  # prevent duplicates
        for out in outputs:
            name = out.get("name", "")
            url = out.get("href", "")
            data_type = out.get("dataType", "")
            content_type = out.get("contentType", "")
            STORE.persist_task_result(
                db_id,
                url,
                name,
                data_type,
                content_type,
                storage_provider="url_file_store",
            )

    task_data.data["minimizer_success"] = True
    task_data.data["task_state"] = "of_cleanup"

    task_data.save(commit=True)

    # start working on cleaning up the open objective function result
    of_cleanup_task = enter_of_cleanup.s(db_id=db_id)
    of_cleanup_task.apply_async()

    # check if optimization is finished
    return self.replace(check_final_result.s(db_id=db_id))


@CELERY.task(
    name=f"{Optimizer.instance.identifier}.handle_of_result",
    bind=True,
    ignore_result=True,
)
def handle_of_result(self, db_id: int):
    TASK_LOGGER.info(f"Start gathering of result for task with db_id={db_id}")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(task_data.data, dict)

    result_url = task_data.data["of_task_url"]

    status, result_data = get_task_result_no_wait(result_url)

    if status == "PENDING":
        return  # result was in fact not settled

    if status == "FAILURE":
        task_data.data["of_success"] = False
        task_data.save(commit=True)
        return self.replace(check_final_result.s(db_id=db_id))

    # TODO: load of data?

    task_data.data["of_success"] = True

    task_data.save(commit=True)

    # check if optimization is finished
    return self.replace(check_final_result.s(db_id=db_id))


@CELERY.task(
    name=f"{Optimizer.instance.identifier}.check_final_result",
    bind=True,
    ignore_result=True,
)
def check_final_result(self, db_id: int):
    TASK_LOGGER.info(f"Checking final result for task with db_id={db_id}")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(task_data.data, dict)

    if task_data.status in ("SUCCESS", "FAILURE"):
        TASK_LOGGER.debug("Task result is already settled.")
        return

    of_state = task_data.data.get("of_success", None)
    minimizer_state = task_data.data.get("minimizer_success", None)

    if of_state is None or minimizer_state is None:
        TASK_LOGGER.debug("One or more sub tasks are not finished.")
        return

    if of_state is False or minimizer_state is False:
        TASK_LOGGER.info("One or more sub tasks have finished as failed!")
        task_data.task_status = "FAILURE"
        task_data.save(commit=True)

        app = current_app._get_current_object()
        TASK_STATUS_CHANGED.send(app, task_id=db_id)
        return

    # both subtasks must have finished successfully
    return self.replace(
        save_task_result.s(task_log="Finished optimization process.", db_id=db_id)
    )
