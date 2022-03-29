import logging
import threading
import requests
from typing import Callable

logger = logging.getLogger(__name__)


def periodic_thread_max_one(interval, fun: Callable, camunda_client, run=False):
    """
    Semi periodic function runner. Each function runs in a thread and every "interval" seconds after(!)
    the last function finished. There can be at most one function running at the same time. Only runs while bpmn model
    instance is active.
    :param interval:
    :param fun: The function to run
    :param run: Used to avoid running the function in main process
    :return:
    """

    if camunda_client.process_end:
        return

    thread = threading.Timer(
        interval,
        periodic_thread_max_one,
        [interval, fun, camunda_client, True]
    )

    thread.daemon = True

    if run:
        fun()

    thread.start()
    return thread


def endpoint_found(response):
    if not response or (response.status_code != 204 and (
            'message' in response.json() and response.json()['message'] == "HTTP 404 Not Found")):
        logger.warning(f"Endpoint not found or failed {response.url}")
        logger.warning(f"{response.json} / {response.text}")
        raise Exception
    else:
        return True


def endpoint_found_simple(response):
    if not response:
        logger.warning("Endpoint not found or failed")
        raise Exception
    else:
        return True


def request_json(url: str):
    response = requests.get(url)
    if endpoint_found(response):
        return response.json()


def get_input_parameters():
    parameters = {}
    text = input("Input plugin parameters as \"key: value\" (type done to submit)\n")
    while text.lower() != "done":
        key, val = text.split(':', 1)
        parameters[key] = val.strip()
        text = input("")
    return parameters
