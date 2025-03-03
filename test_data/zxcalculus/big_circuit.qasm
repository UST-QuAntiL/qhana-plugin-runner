OPENQASM 2.0;
include "qelib1.inc";

qreg q[7];
creg c[4];
h q[0];
h q[2];
h q[4];
swap q[5], q[6];
cx q[0], q[2];
ccx q[2], q[3], q[4];
measure q[0] -> c[0];
cx q[1], q[2];
measure q[4] -> c[3];
h q[1];
ccx q[1], q[5], q[6];
measure q[5] -> c[3];
measure q[1] -> c[1];
measure q[3] -> c[3];