This folder contains plugins that use Qiskit for machine learning.

The following dependencies are used by these plugins:
- qiskit~=2.3.0
- qiskit-machine-learning~=0.9.0
- scikit-learn~=1.8.0
- plotly~=6.6.0
- pandas~=2.3.3
- muid~=0.5.3

## Compatibility

`compat.py` (`ensure_qiskit_machine_learning_compat`) monkey-patches
`qiskit.primitives` so `qiskit-machine-learning` keeps working on `qiskit>=2`.
It is imported by the SVM, qiskit quantum kernel estimation, and VQC plugins.