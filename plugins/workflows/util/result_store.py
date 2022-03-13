import logging
from typing import List

import requests

from ..datatypes.qhana_datatypes import QhanaResult
from ..util.helper import endpoint_found_simple

logger = logging.getLogger(__name__)


class ResultStore:
    def __init__(self):
        self.store: List[QhanaResult] = []

    def store_result(self, qhana_result: QhanaResult):
        """
        Store the results from the QHAna plugin
        :param qhana_result: result from running the plugin
        :return:
        """
        self.store.append(qhana_result)

    def print_to_console(self):
        """
        Print all results in the ResultStore
        :return:
        """
        logger.info("RESULT_STORE (from oldest to newest):")
        for i, qhana_result in enumerate(self.store):
            external_task = qhana_result.qhana_task.external_task

            logger.info("{}. External Task ID: {} \t Qhana Task ID: {} \t Qhana Plugin: {}"
                        .format(i+1, external_task.id, qhana_result.qhana_task.id, qhana_result.qhana_task.plugin.name))
            for j, output in enumerate(qhana_result.output_list):
                response = requests.get(output.href)
                if endpoint_found_simple(response):
                    response = response.text
                logger.info("\t {}. Name: {} \t Content Type: {} \t Data Type: {} \t Resource: {}"
                            .format(j+1, output.name, output.content_type, output.data_type, output.href))
                logger.info("\t\t {}".format(response))

