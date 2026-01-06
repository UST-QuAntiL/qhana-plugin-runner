from . import plugin

try:
    from . import routes, pattern_atlas_dynamic, foo
except ImportError as err:
    import traceback

    traceback.print_exception(err)
