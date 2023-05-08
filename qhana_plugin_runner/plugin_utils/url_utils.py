import urllib.parse


def get_plugin_name_from_plugin_url(plugin_url: str) -> str:
    decoded: str = urllib.parse.unquote(plugin_url).rstrip("/")
    plugin_name: str = decoded.split("/")[-1]
    return plugin_name
