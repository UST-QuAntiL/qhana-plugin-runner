OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[4];
h q[0];
cx q[0], q[1];
h q[1];
cx q[0], q[1];
x q[0];
h q[1];
h q[0];
swap q[0], q[1];
h q[0];
cx q[0], q[1];
h q[1];
x q[1];
cx q[0], q[1];
h q[0];
s q[1];
t q[0];
z q[0];
cx q[0], q[1];
h q[1];
measure q[1] -> c[1];
measure q[0] -> c[0];
