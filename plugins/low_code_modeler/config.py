from flask import Flask
from qhana_plugin_runner.registry_client.client import PluginRegistryClient


DEFAULT_CONFIG = {
    "NisqAnalyzerEndpoint": "http://localhost:8098/nisq-analyzer",
    "QunicornEndpoint": "http://localhost:8080",
    "LowcodeBackendEndpoint": "http://localhost:8000",
    "PatternAtlasUiEndpoint": "http://localhost:1978",
    "PatternAtlasApiEndpoint": "http://localhost:1977/patternatlas/patternLanguages/af7780d5-1f97-4536-8da7-4194b093ab1d",
    "QcAtlasEndpoint": "http://localhost:6626",
    "GithubRepositoryOwner": "",
    "GithubRepositoryName": "",
    "GithubBranch": "",
    "GithubToken": "",
}


def get_config(app: Flask) -> dict[str, str]:
    with PluginRegistryClient(app) as client:
        config = dict(DEFAULT_CONFIG)
        services = client.fetch_by_rel(
            ["service"], {"service-id": ",".join(config.keys())}
        )
        if services is not None:
            for item in services.data.get("items", []):
                resource_key = item.get("resourceKey", {"serviceId": None})
                service_id = resource_key.get("serviceId")
                if service_id is None:
                    continue
                href = item.get("href")
                config[service_id] = href
        return config
