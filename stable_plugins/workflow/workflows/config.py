import uuid
from os import PathLike, environ
from pathlib import Path
from typing import Sequence, Tuple

from flask import Flask
from typing_extensions import Required, TypedDict


class FormInputConfig(TypedDict):
    file_url_prefix: str
    choice_value_prefix: str
    enum_value_prefix: str
    value_separator: str
    text_input_mode: str
    filename_input_mode: str
    datatype_input_mode: str


class WorkflowConfig(TypedDict):
    workflow_error_prefix: str
    legacy_plugin_task_topic_prefix: str
    legacy_step_topic_prefix: str
    disable_legacy_support: bool
    qhana_task_topic: str
    plugin_variable: str
    plugin_version_variable: str
    step_variable: str
    task_input_variable_prefix: str
    task_output_variable_prefix: str
    return_variable_prefix: str
    prefix_separator: str
    form_conf: FormInputConfig


class WorkflowPluginConfig(TypedDict, total=False):
    camunda_base_url: Required[str]
    worker_id: Required[str]
    request_timeout: Required[float]
    max_concurrent_tasks: Required[int]
    camunda_queue_polling_rate: Required[float]
    task_result_polling_rate: Required[float]
    workflow_conf: Required[WorkflowConfig]
    workflow_folder: PathLike


def separate_prefixes(name: str, conf: WorkflowConfig) -> Tuple[str, Sequence[str]]:
    """Split a name into prefixes and name portion based on the current config."""
    sep = conf["prefix_separator"]
    prefixes = {
        conf["return_variable_prefix"],
        conf["task_input_variable_prefix"],
        conf["task_output_variable_prefix"],
    }
    split_name = name.split(sep)
    name_start = 0
    while split_name[name_start] in prefixes:
        name_start += 1
    return sep.join(split_name[name_start:]), split_name[:name_start]


def get_config(app: Flask | None = None) -> WorkflowPluginConfig:
    app_config = app.config if app else {}

    workflow_folder = Path(
        app_config.get(
            "WORKFLOW_FOLDER",
            environ.get("WORKFLOW_FOLDER", "./workflows"),
        )
    )
    if not workflow_folder.is_absolute() and app is not None:
        workflow_folder = Path(app.instance_path) / workflow_folder
        workflow_folder = workflow_folder.resolve()

    camunda_url = app_config.get(
        "CAMUNDA_API_URL",
        environ.get("CAMUNDA_API_URL", "http://localhost:8080/engine-rest"),
    )

    default_timout: str = app_config.get(
        "REQUEST_TIMEOUT",
        environ.get("REQUEST_TIMEOUT", str(5 * 60)),
    )
    timout_int = 5 * 60
    if default_timout.isdigit():
        timout_int = int(default_timout)

    max_parrallelism: str = app_config.get(
        "EXTERNAL_TASK_CONCURRENCY",
        environ.get("EXTERNAL_TASK_CONCURRENCY", str(10)),
    )
    max_parrallelism_int = 10
    if max_parrallelism.isdigit():
        max_parrallelism_int = int(max_parrallelism)

    worker_id: str = app_config.get(
        "CAMUNDA_WORKER_ID",
        environ.get("CAMUNDA_WORKER_ID", str(uuid.uuid4())),
    )

    workflow_form_config: FormInputConfig = {
        "file_url_prefix": "file_url",
        "choice_value_prefix": "choice",
        "enum_value_prefix": "enum",
        "value_separator": "::",
        "datatype_input_mode": "dataType",
        "filename_input_mode": "name",
        "text_input_mode": "plain",
    }

    workflow_conf: WorkflowConfig = {
        "workflow_error_prefix": "qhana",
        "disable_legacy_support": False,
        "legacy_plugin_task_topic_prefix": "plugin",
        "legacy_step_topic_prefix": "plugin-step",
        "qhana_task_topic": "qhana-task",
        "plugin_variable": "qhanaPlugin",
        "plugin_version_variable": "qhanaPluginVersion",
        "step_variable": "qhanaPluginStep",
        "task_output_variable_prefix": "qoutput",
        "task_input_variable_prefix": "qinput",
        "prefix_separator": ".",
        "return_variable_prefix": "return",
        "form_conf": workflow_form_config,
    }

    return {
        "camunda_base_url": camunda_url,
        "worker_id": worker_id,
        "request_timeout": timout_int,
        "camunda_queue_polling_rate": 5.0,
        "task_result_polling_rate": 5.0,
        "max_concurrent_tasks": max_parrallelism_int,
        "workflow_folder": workflow_folder,
        "workflow_conf": workflow_conf,
    }
