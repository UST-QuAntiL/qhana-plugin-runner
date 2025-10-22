from flask import Flask
from qhana_plugin_runner.registry_client.client import PluginRegistryClient


DEFAULT_CONFIG = {
    "NisqAnalyzerEndpoint": "http://localhost:8098/nisq-analyzer",
    "QunicornEndpoint": "http://localhost:8081",
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
            for api_link in services.data.get("items", []):
                service = client.fetch_by_api_link(api_link)
                service_id = service.data.get("serviceId")
                url = service.data.get("url")
                if service_id is None or url is None:
                    continue
                config[service_id] = url
        return config
