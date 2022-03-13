import http.client
import logging
import re
from threading import Timer
from typing import Callable, Any, List, TYPE_CHECKING, Optional

import requests

from ..datatypes.camunda_datatypes import ExternalTask
from ..datatypes.qhana_datatypes import QhanaResult, QhanaPlugin, QhanaTask, QhanaOutput
from ..util.helper import endpoint_found, periodic_thread_max_one

logger = logging.getLogger(__name__)


class QhanaPluginEndListener:
    """
    Listens for QHAna plugins which have finished
    """

    def __init__(
            self,
            camunda_client,
            callback: Callable[[..., List[QhanaResult]], Any],
    ):
        self.camunda_client = camunda_client
        self.thread: Optional[Timer] = None
        self.callback = callback
        self.tasks: List[QhanaTask] = []

    def add_qhana_task(self, external_task: ExternalTask, plugin: QhanaPlugin, params):
        """
        Starts a QHAna task
        :param external_task:
        :param plugin: The plugin to run
        :param params: Parameters for running the plugin
        :return:
        """
        response = requests.post(plugin.process_endpoint, data=params)
        url = response.url
        if response.status_code == http.client.OK:
            response = response.json()
            db_id = re.search("/\d+/", url).group(0)[1:-1]
            self.tasks.append(QhanaTask.deserialize(response, db_id, external_task, plugin))
            logger.info(f"Started QHAna plugin {plugin.identifier}")
        elif response.status_code == http.client.UNPROCESSABLE_ENTITY:
            logger.warning(f"Received unprocessable entity on endpoint {url}")
            self.camunda_client.external_task_bpmn_error(external_task, "qhana-unprocessable-entity-error",
                                                         "Plugin invocation received unprocessable entities and could "
                                                         "not proceed.")

    def start(self):
        """
        Begin listening for finished qhana plugin instances
        :return:
        """
        self.thread = periodic_thread_max_one(self.camunda_client.m_poll_interval, self.poll, self.camunda_client)
        logger.info("Started new qhana plugin end listener")
        return self

    def poll(self):
        """
        Poll for finished qhana plugin instances
        :return:
        """
        qhana_task_results = []
        for qhana_task in self.tasks:
            response = requests.get(f"{qhana_task.plugin.api_endpoint}/tasks/{qhana_task.id}")
            if endpoint_found(response):
                contents = response.json()
                if contents["status"] == "SUCCESS":
                    outputs: List[QhanaOutput] = []
                    for output in contents["outputs"]:
                        outputs.append(QhanaOutput.deserialize(output))
                    qhana_result = QhanaResult(qhana_task, outputs)
                    self.tasks.remove(qhana_task)
                    qhana_task_results.append(qhana_result)
                elif contents["status"] == "FAILURE":
                    logger.warning(f"Qhana task with id: {qhana_task.id} failed. Throwing bpmn exception..")
                    self.tasks.remove(qhana_task)
                    self.camunda_client.external_task_bpmn_error(qhana_task.external_task, "qhana-plugin-failure",
                                                                 "Plugin failed execution.")

        if qhana_task_results:
            logger.info(f"Found {len(qhana_task_results)} new qhana task results")
            self.callback(self.camunda_client, qhana_task_results)
