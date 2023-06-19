# public .env file used by the flask command use only during development
# to override values in here locally use a private .env file
FLASK_APP=qhana_plugin_runner
FLASK_DEBUG=true # set to production if in production!

# configure default port
FLASK_RUN_PORT=5005

# plugin folders that should be loaded by default
PLUGIN_FOLDERS=./plugins:./stable_plugins/classical_ml/data_preparation:./stable_plugins/classical_ml/scikit_ml:./stable_plugins/data_synthesis:./stable_plugins/demo:./stable_plugins/file_utils:./stable_plugins/infrastructure:./stable_plugins/muse:./stable_plugins/nisq_analyzer:./stable_plugins/quantum_ml/max_cut:./stable_plugins/quantum_ml/pennylane_qiskit_ml:./stable_plugins/quantum_ml/qiskit_ml:./stable_plugins/visualization/complex:./stable_plugins/visualization/file_types:./stable_plugins/workflow
