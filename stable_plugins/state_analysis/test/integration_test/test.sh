#!/bin/bash
set -e  # Exit script on error

########################################
# 1. Conda Environment Setup
########################################

# Name of the new Anaconda environment and Python version
CONDA_ENV_NAME="qhana-clean-test-env"
PYTHON_VERSION="3.10"

echo "Initializing Conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"

# If the environment already exists, remove it
if conda info --envs | grep -q "$CONDA_ENV_NAME"; then
    echo "Environment '$CONDA_ENV_NAME' already exists. Removing it..."
    conda env remove -n "$CONDA_ENV_NAME" -y
fi

# Create a new environment
echo "Creating a new Anaconda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
conda create --name "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y

# Activate environment
echo "Activating Anaconda environment '$CONDA_ENV_NAME'..."
conda activate "$CONDA_ENV_NAME"


########################################
# 2. Determine Project Root & Poetry Setup
########################################

# Save the current directory (should be qhana-plugin-runner/stable_plugins/state_analysis/test/integration_test)
ORIG_DIR=$(pwd)

# Determine project root (qhana-plugin-runner) – move up four levels from the current folder
QPR_ROOT=$(cd ../../../.. && pwd)
echo "Project root determined as: $QPR_ROOT"

echo "Initializing qhana-plugin-runner..."
cd "$QPR_ROOT" || { echo "Directory qhana-plugin-runner not found!"; exit 1; }

# Install project dependencies with Poetry
echo "Installing project dependencies with Poetry..."
poetry install

# Check if Qiskit is installed
echo "Checking if Qiskit is installed..."
poetry run python -c "from qiskit import __version__; print(f'Qiskit is installed. Version: {__version__}')" || {
    echo "Error: Qiskit is not installed or not accessible in the environment."
    exit 1
}

########################################
# 3. Set PYTHONPATH & Run Tests with Coverage
########################################

# Set PYTHONPATH so that the state_analysis folder is importable
# (state_analysis contains the "common" subfolder required by the test)
QSA_PATH="$QPR_ROOT/stable_plugins/state_analysis"
export PYTHONPATH="$QSA_PATH"
echo "Set PYTHONPATH to: $PYTHONPATH"

# Return to the original test directory
cd "$ORIG_DIR" || { echo "Cannot return to test directory"; exit 1; }

# Run tests 
echo "Running tests with coverage..."
poetry run pytest  -v --tb=long test_swaptest_quantum_orth_api_qasm.py test_lin_dep_inhx_api_vector.py test_lin_dep_inhx_api_qasm.py  test_generate_one_circuit_with_multiple_states.py test_generate_one_circuit_with_two_states.py test_lin_dep_api_vector.py test_schmidt_api_qasm.py test_schmidt_api_vector.py test_po_api_qasm.py test_po_api_vector.py test_opr_api_qasm.py test_opr_api_vector.py test_orth_api_vector.py test_orth_api_qasm.py test_file_maker_api.py test_lin_dep_api_qasm.py 

########################################
# 4. Deactivate Conda Environment
########################################
echo "Deactivating Conda environment..."
conda deactivate