import logging
from threading import Timer
from typing import Callable, Optional

import requests

from ..datatypes.camunda_datatypes import ProcessInstance
from ..util.helper import periodic_thread_max_one
from ..util.helper import endpoint_found

logger = logging.getLogger(__name__)


class CamundaProcessEndListener:
    """
    Listens for camunda process instance end
    """

    def __init__(
            self,
            camunda_client,
            callback: Callable,
    ):
        self.camunda_client = camunda_client
        self.thread: Optional[Timer] = None
        self.callback = callback

    def start(self):
        """
        Begin listening for the camunda process instance end
        :return:
        """
        self.thread = periodic_thread_max_one(self.camunda_client.m_poll_interval, self.poll, self.camunda_client)
        logger.info("Started new camunda process end listener")
        return self

    def poll(self):
        """
        Poll for the camunda process instance end
        :return:
        """
        response = requests.get(f"{self.camunda_client.m_base_url}/process-instance")
        if endpoint_found(response):
            for process in response.json():
                if process["id"] == self.camunda_client.process_instance.id:
                    self.camunda_client.process_instance = ProcessInstance.deserialize(process)
                    return

            self.callback()
