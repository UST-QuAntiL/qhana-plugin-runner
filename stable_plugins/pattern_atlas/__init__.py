from . import plugin

try:
    from . import routes, pattern_atlas_dynamic
except ImportError:
    # fails during `poetry run flask install`, before httpx (plugin dependency) is installed
    pass
