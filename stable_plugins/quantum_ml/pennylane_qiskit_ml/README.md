This folder contains plugins that implement machine learning algorithms with pennylane and qiskit.

The following dependencies are used by these plugins:
- matplotlib~=3.5.1
- qiskit~=2.3.0
- pennylane~=0.44.0
- pennylane-qiskit~=0.44.1
- scikit-learn~=1.8.0
- torch~=2.10.0
- muid~=0.5.3

## Compatibility

`pennylane_qiskit_compat.py` provides two monkey-patches used by the QCNN,
QNN, hybrid autoencoder, and other pennylane-qiskit plugins on `qiskit>=2`:

- `ensure_qiskit_ibm_provider_compat()` re-creates symbols that
  `qiskit-ibm-provider` and `qiskit-ibm-runtime` rely on (`ProviderV1`,
  `BackendV1`, etc.) when those packages have not been updated for
  `qiskit>=2`.
- `pennylane_qiskit_version_override()` temporarily overrides
  `qiskit.__version__` so `pennylane-qiskit` accepts the newer qiskit
  release while still pinning to a tested range.

