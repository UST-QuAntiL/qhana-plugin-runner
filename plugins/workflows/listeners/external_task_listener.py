from __future__ import annotations
import logging
from threading import Timer
from typing import Callable, Any, List, TYPE_CHECKING, Optional

import requests

from ..datatypes.camunda_datatypes import ExternalTask
from ..util.helper import periodic_thread_max_one, endpoint_found

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..clients.camunda_client import CamundaClient


class ExternalTaskListener:
    """
    Listens for new camunda external tasks available
    """

    def __init__(
            self,
            camunda_client: CamundaClient,
            callback: Callable[[..., List[ExternalTask]], Any],
    ):
        self.camunda_client = camunda_client
        self.thread: Optional[Timer] = None
        self.callback = callback

    def start(self):
        """
        Begin listening for new camunda external tasks
        :return:
        """
        self.thread = periodic_thread_max_one(self.camunda_client.m_poll_interval, self.poll, self.camunda_client)
        logger.info("Started new camunda external task listener")
        return self

    def poll(self):
        """
        Poll for new camunda external tasks
        :return:
        """
        response = requests.get(f"{self.camunda_client.m_base_url}/external-task")
        if endpoint_found(response):
            external_tasks = response.json()
            tasks = []

            for external_task in external_tasks:
                try:
                    execution_id = self.camunda_client.get_task_execution_id(external_task["id"])
                except:
                    # TODO: Tidy up... (Sometimes a task is completed but it takes a bit for camunda to remove it
                    #  from the external task queue, thus we end up here)
                    continue

                external_task["executionId"] = execution_id
                external_task = ExternalTask.deserialize(external_task)
                task_topic = external_task.topic_name

                if external_task.process_instance_id == self.camunda_client.process_instance.id \
                        and task_topic.startswith(self.camunda_client.plugin_prefix + "."):
                    self.camunda_client.lock(external_task)
                    tasks.append(external_task)

            if tasks:
                # Create QHAna plugin instances
                self.callback(self.camunda_client, tasks)
