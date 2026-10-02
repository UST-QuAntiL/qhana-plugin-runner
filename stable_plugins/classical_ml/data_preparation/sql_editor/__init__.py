from .plugin import SQLEditor

_plugin_name = SQLEditor.name
__version__ = SQLEditor.version

try:
    from . import routes  # noqa: F401,E402
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
