#
# Endpoints
# TODO: Remove and replace references with already existing config in qhana-plugin-runner
#
CAMUNDA_BASE_URL = "http://localhost:8080/engine-rest"
QHANA_PLUGIN_ENDPOINTS = ["http://localhost:5005/"]

#
# Interval lengths and polling timeouts
#
CAMUNDA_GENERAL_POLL_TIMEOUT = 5.0
EXTERNAL_WATCHER_INTERVAL = 5.0

#
# QHAna
#
QHANA_INPUT_PREFIX = "qinput"
QHANA_INPUT_MODE_TEXT = "plain"
QHANA_INPUT_MODE_FILENAME = "name"
QHANA_INPUT_MODE_DATATYPE = "dataType"
