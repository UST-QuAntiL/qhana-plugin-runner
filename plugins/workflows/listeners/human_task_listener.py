from __future__ import annotations
import logging
from threading import Timer
from typing import Callable, Any, List, TYPE_CHECKING, Optional

import requests
from flask import url_for

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from ..datatypes.camunda_datatypes import ExternalTask, HumanTask
from ..util.helper import periodic_thread_max_one, endpoint_found
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..clients.camunda_client import CamundaClient


class HumanTaskListener:

    def __init__(
        self,
        db_id,
        task_data: ProcessingTask,
        camunda_client: CamundaClient,
    ):
        self.db_id = db_id
        self.task_data = task_data
        self.camunda_client = camunda_client
        self.thread: Optional[Timer] = None
        self.current_human_task = None

    def start(self):
        """
        Begin listening for new camunda human tasks
        :return:
        """
        self.thread = periodic_thread_max_one(self.camunda_client.m_poll_interval, self.poll, self.camunda_client)
        logger.info("Started new camunda human task listener")
        return self

    def poll(self):
        """
        Poll for new camunda human tasks
        :return:
        """
        response = requests.get(f"{self.camunda_client.m_base_url}/task")
        if endpoint_found(response):
            human_tasks = response.json()

            if self.current_human_task is None:
                for human_task in human_tasks:
                    human_task = HumanTask.deserialize(human_task)

                    if (human_task.delegation_state == "PENDING" or human_task.delegation_state is None) and \
                            human_task.process_instance_id == self.camunda_client.process_instance.id:
                        # rendered_form = self.camunda_client.get_human_task_rendered_form(human_task.id)
                        # logger.info(f"Rendered human task form: \n {rendered_form}")
                        #
                        form_variables = requests.get(f"{self.camunda_client.m_base_url}/task/{human_task.id}/form-variables")
                        form_variables = str(form_variables.json())
                        #
                        logger.info(f"{form_variables}")
                        #
                        # task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=self.db_id)
                        # task_data.data["form_params"] = str(form_variables.json())
                        # task_data.save(commit=True)
                        #
                        # href = f"http://localhost:5005/plugins/workflows@v0-1-1/{self.db_id}/demo-step-process/"
                        # ui_href = f"http://localhost:5005/plugins/workflows@v0-1-1/{self.db_id}/demo-step-ui/"
                        #
                        # task = add_step.s(
                        #     db_id=self.db_id, step_id=human_task.id, href=href, ui_href=ui_href, prog_value=50, task_log="",
                        # )
                        #
                        # task.link_error(save_task_error.s(db_id=self.db_id))
                        # task.apply_async()
                        #
                        # logger.info("Started step..")
                        self.current_human_task = human_task
                        return
