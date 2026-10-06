This folder contains plugins that implement machine learning algorithms with pennylane and qiskit.

The following dependencies are used by these plugins:
- matplotlib~=3.5.1
- qiskit~=2.3.0
- pennylane~=0.44.0
- pennylane-qiskit~=0.44.1
- scikit-learn~=1.8.0
- torch~=2.10.0
- muid~=0.5.3

## Known limitations

The `ibmq_*` and `custom_ibmq` backends of these plugins do not work with the
pinned dependency versions and raise an error when selected. They are built on
`qiskit-ibm-provider` and the `qiskit.ibmq` pennylane device, both of which were
removed upstream:

- `qiskit-ibm-provider` is deprecated and not compatible with `qiskit>=2`. It is
  replaced by `qiskit-ibm-runtime`.
- `pennylane-qiskit~=0.44.1` no longer registers a `qiskit.ibmq` device. Remote
  IBM backends are reached through its `qiskit.remote` device instead.

Porting these backends to `qiskit-ibm-runtime` and `qiskit.remote` is out of
scope for the qiskit 2 update. The local `aer_*` and
`pennylane_default.qubit` backends are unaffected and work as before.
