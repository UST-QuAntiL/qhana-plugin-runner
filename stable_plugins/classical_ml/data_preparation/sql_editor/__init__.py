from .plugin import SQLEditor

_plugin_name = SQLEditor.name
__version__ = SQLEditor.version

from . import routes  # noqa: F401,E402
