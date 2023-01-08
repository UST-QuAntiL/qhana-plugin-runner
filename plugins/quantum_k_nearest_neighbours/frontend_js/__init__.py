import os

_file_dir = os.path.dirname(__file__)

_frontend_js_content = ""
with open(_file_dir + "/frontend.js") as f:
    _frontend_js_content = f.read()
    f.close()


def get_frontend_js() -> str:
    return _frontend_js_content
