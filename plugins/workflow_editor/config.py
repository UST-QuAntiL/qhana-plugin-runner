from os import environ
from typing import Optional
from urllib.parse import urljoin
from uuid import uuid4

from flask import Flask

from qhana_plugin_runner.registry_client.client import PluginRegistryClient

CONFIG_KEY = "WF_EDITOR_CONFIG"

ENV_VAR_PREFIX = "WF_EDITOR_"

WF_STATE_KEY = "WORKFLOWS"

SERVICE_CONFIG_VARS = {
    "PLUGIN_REGISTRY_URL": "qhana-plugin-registry",
    "OPENTOSCA_ENDPOINT": "opentosca",
    "WINERY_ENDPOINT": "winery",
    "PATTERN_ATLAS_ENDPOINT": "pattern-atlas",
    "PATTERN_ATLAS_UI_ENDPOINT": "pattern-atlas-ui",
    "QC_ATLAS_ENDPOINT": "qc-atlas",
    "NISQ_ANALYZER_ENDPOINT": "nisq-analyzer",
    "NISQ_ANALYZER_UI_ENDPOINT": "nisq-analyzer-ui",
    "QPROV_ENDPOINT": "qprov",
    "SCRIPT_SPLITTER_ENDPOINT": "script-splitter",
    "QISKIT_RUNTIME_HANDLER_ENDPOINT": "qiskit-runtime-handler",
    "AWS_RUNTIME_HANDLER_ENDPOINT": "aws-runtime-handler",
    "TRANSFORMATION_FRAMEWORK_ENDPOINT": "transformation-framework",
    "CAMUNDA_ENDPOINT": "camunda",
    "QHANA_BACKEND": "qhana-backend",
}

WORKFLOW_CONFIG_VARS = (
    "SERVICE_DATA_CONFIG",
    "SCRIPT_SPLITTER_THRESHOLD",
    "DOWNLOAD_FILE_NAME",
    "TRANSFORMED_WORKFLOW_HANDLER",
    "AUTO_SAVE_FILE_OPTION",
    "FILE_NAME",
    "FILE_FORMAT",
    "AUTO_SAVE_INTERVAL",
    "GITHUB_TOKEN",
    "QRM_REPONAME",
    "QRM_USERNAME",
    "QRM_REPOPATH",
    "UPLOAD_GITHUB_REPO",
    "UPLOAD_GITHUB_USER",
    "UPLOAD_GITHUB_REPOPATH",
    "UPLOAD_FILE_NAME",
    "UPLOAD_BRANCH_NAME",
)


def load_config_from_env(app: Flask):
    # config values for service endpoints
    for var in SERVICE_CONFIG_VARS:
        if app.config.get(var):
            continue
        if var in environ:
            app.config[var] = environ[var]

    # workflow editor specific config values
    wf_editor_config = app.config.get("WF_EDITOR", {})
    app.config["WF_EDITOR"] = wf_editor_config
    for var in WORKFLOW_CONFIG_VARS:
        env_var = f"{ENV_VAR_PREFIX}{var}"
        if env_var in environ:
            wf_editor_config[var] = environ[env_var]


def get_config_from_app(app: Optional[Flask]) -> dict:
    config = {
        "PLUGIN_REGISTRY_URL": "http://localhost:5006/api",
        "SERVICE_DATA_CONFIG": "http://localhost:8000/service-task",
        "OPENTOSCA_ENDPOINT": "http://localhost:1337/csars",
        "WINERY_ENDPOINT": "http://localhost:8080/winery",
        "PATTERN_ATLAS_ENDPOINT": "http://localhost:8080/pattern-atlas",
        "PATTERN_ATLAS_UI_ENDPOINT": "http://localhost:8080/pattern-atlas",
        "QC_ATLAS_ENDPOINT": "http://localhost:8080/qc-atlas",
        "NISQ_ANALYZER_ENDPOINT": "http://localhost:8080/nisq-analyzer",
        "NISQ_ANALYZER_UI_ENDPOINT": "http://localhost:8080/nisq-analyzer",
        "QPROV_ENDPOINT": "http://localhost:8080/qprov",
        "SCRIPT_SPLITTER_ENDPOINT": "http://localhost:8080/script-splitter",
        "SCRIPT_SPLITTER_THRESHOLD": "0.5",
        "QISKIT_RUNTIME_HANDLER_ENDPOINT": "http://localhost:8080/qiskit-runtime-handler",
        "AWS_RUNTIME_HANDLER_ENDPOINT": "http://localhost:8080/aws-runtime-handler",
        "TRANSFORMATION_FRAMEWORK_ENDPOINT": "http://localhost:8080/transformation-framework",
        "CAMUNDA_ENDPOINT": "http://localhost:8080/camunda",
        "DOWNLOAD_FILE_NAME": "workflow.bpmn",
        "TRANSFORMED_WORKFLOW_HANDLER": "inline",
        "AUTO_SAVE_FILE_OPTION": "interval",
        "FILE_NAME": "quantum-workflow",
        "FILE_FORMAT": "bpmn",
        "AUTO_SAVE_INTERVAL": "300000",
        "GITHUB_TOKEN": "",
        "QRM_REPONAME": "",
        "QRM_USERNAME": "",
        "QRM_REPOPATH": "",
        "UPLOAD_GITHUB_REPO": "",
        "UPLOAD_GITHUB_USER": "",
        "UPLOAD_GITHUB_REPOPATH": "qrms",
        "UPLOAD_FILE_NAME": "quantum-workflow-model",
        "UPLOAD_BRANCH_NAME": "",
        "camunda_worker_id": app.config.get(
            "CAMUNDA_WORKER_ID",
            environ.get("CAMUNDA_WORKER_ID", str(uuid4())),
        ),
    }
    if not app:
        return config
    if "PLUGIN_REGISTRY_URL" in app.config:
        config["PLUGIN_REGISTRY_URL"] = app.config["PLUGIN_REGISTRY_URL"]
    for var in SERVICE_CONFIG_VARS:
        if var in app.config:
            config[var] = app.config[var]
    config.update(app.config.get("WF_EDITOR", {}))
    return config


def get_config_from_registry(app: Optional[Flask]):
    config = {}
    if not app:
        return config
    registry_client = PluginRegistryClient(app)
    with registry_client as client:
        # load service endpoints from registry
        service_id_to_conf = {s: var for var, s in SERVICE_CONFIG_VARS.items()}
        services = client.fetch_by_rel(
            ["service"], {"service-id": ",".join(service_id_to_conf.keys())}
        )
        service_list = services.data.get("items", []) if services else []
        for service_link in service_list:
            service = client.fetch_by_api_link(service_link)
            if service:
                service_id = service.data["serviceId"]
                config[service_id_to_conf[service_id]] = service.data["url"]

        # load env vars from registry
        env_response = client.fetch_by_rel(["env"])
        env_list = env_response.data.get("items", []) if env_response else []
        for env_link in env_list:
            env_var = env_link.get("resourceKey", {}).get("envId", "")
            env_var = env_var.removeprefix(ENV_VAR_PREFIX)
            if env_var not in WORKFLOW_CONFIG_VARS:
                continue
            value = client.fetch_by_api_link(env_link)
            if value:
                config[env_var] = value.data["value"]

    return config


def postprocess_config(config: dict) -> dict:
    camunda: str = config["CAMUNDA_ENDPOINT"]
    if not camunda.endswith("/engine-rest"):
        if not camunda.endswith("/"):
            camunda = camunda + "/"
        config["CAMUNDA_ENDPOINT"] = urljoin(camunda, "./engine-rest")
    if not config["FILE_NAME"]:
        config["FILE_NAME"] = "quantum-workflow"
    patternatlas: str = config["PATTERN_ATLAS_ENDPOINT"]
    if not patternatlas.endswith("/"):
        patternatlas = patternatlas + "/"
    config["PATTERN_ATLAS_ENDPOINT"] = urljoin(
        patternatlas, "./patternLanguages/af7780d5-1f97-4536-8da7-4194b093ab1d"
    )
    return config
