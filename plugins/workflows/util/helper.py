import logging
import re

import requests
from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)


def get_form_variables(rendered_form, instance_variables):
    form_variables = form_variables_from_html(rendered_form)
    variables = instance_variables

    for k in variables.copy().keys():
        if k not in form_variables:
            del variables[k]

    return variables


def form_variables_from_html(html):
    variables = []
    matches = re.findall(r'<input class="[^"]*" name="[^"]*"', html)
    for match in matches:
        variable_name = re.search(r'"[^"]*"$', match).group()[1:-1]
        variables.append(variable_name)

    return variables


def endpoint_found(response):
    if not response or (response.status_code != 204 and (
            'message' in response.json() and response.json()['message'] == "HTTP 404 Not Found")):
        TASK_LOGGER.warning(f"Endpoint not found or failed {response.url}")
        TASK_LOGGER.warning(f"{response.json} / {response.text}")
        raise Exception
    else:
        return True


def endpoint_found_simple(response):
    if not response:
        TASK_LOGGER.warning("Endpoint not found or failed")
        raise Exception
    else:
        return True


def request_json(url: str):
    response = requests.get(url)
    if endpoint_found(response):
        return response.json()
