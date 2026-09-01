This folder contains plugins that use Qiskit for machine learning.

The following dependencies are used by these plugins:
- qiskit~=2.3.0
- qiskit-machine-learning~=0.9.0
- scikit-learn~=1.8.0
- plotly~=6.6.0
- pandas~=2.3.3
- muid~=0.5.3

## Compatibility

`qiskit-machine-learning~=0.9.0` supports `qiskit>=2` natively, so these plugins
need no compatibility shims. Earlier revisions of this branch monkey-patched the
removed qiskit V1 primitives (`qiskit.primitives.Sampler`/`BaseSampler`) back
into place; that is no longer required and has been removed.

Fidelity kernels are sampled with `BackendSamplerV2` on a local `AerSimulator`
rather than with qiskit's `StatevectorSampler` reference primitive, which
simulates the full statevector in python and becomes exponentially slower as the
qubit count grows.
