"""Validate qsim against Qiskit's Statevector.

Qiskit labels qubit 0 as the least significant bit, while qsim uses qubit 0 as
the most significant bit, so Qiskit circuits are built on reversed wires.
"""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qsim import Circuit
from qsim.algorithms import bell_state, ghz_state, teleportation, grover


def to_qiskit(circ: Circuit) -> QuantumCircuit:
    n = circ.n
    qc = QuantumCircuit(n)
    r = lambda w: n - 1 - w
    for name, m, wires in circ.ops:
        # Qiskit treats the first listed qubit as least significant, so the
        # wires are passed in reverse to keep qsim's first wire most significant.
        qc.unitary(m, [r(w) for w in reversed(wires)], label=name)
    return qc


def assert_same(circ):
    ours = circ.statevector()
    # qsim wire w maps to Qiskit wire n-1-w, so basis-state indices line up
    theirs = Statevector.from_instruction(to_qiskit(circ)).data
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_bell():
    psi = bell_state().statevector()
    np.testing.assert_allclose(psi, np.array([1, 0, 0, 1]) / np.sqrt(2), atol=1e-12)
    assert_same(bell_state())


def test_ghz():
    assert_same(ghz_state(4))


def test_native_qiskit_bell():
    """Compare with a circuit written directly in Qiskit gates."""
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.cx(1, 0)  # qiskit wire 1 == qsim wire 0
    np.testing.assert_allclose(bell_state().statevector(), Statevector.from_instruction(qc).data, atol=1e-12)


@pytest.mark.parametrize("theta,phi", [(0.3, 1.1), (1.7, -0.4), (np.pi, 0.0)])
def test_teleportation(theta, phi):
    circ = teleportation(theta, phi)
    assert_same(circ)
    # q2's reduced state must equal the input state
    psi = circ.statevector().reshape(2, 2, 2)
    rho = np.einsum("abi,abj->ij", psi, psi.conj())
    target = Circuit(1).ry(theta, 0).rz(phi, 0).statevector()
    np.testing.assert_allclose(rho, np.outer(target, target.conj()), atol=1e-10)


@pytest.mark.parametrize("marked", ["101", "0110"])
def test_grover(marked):
    circ = grover(len(marked), marked)
    assert_same(circ)
    p = circ.probabilities()
    assert np.argmax(p) == int(marked, 2)
    assert p[int(marked, 2)] > 0.9


def test_random_circuit():
    rng = np.random.default_rng(7)
    c = Circuit(4)
    for _ in range(40):
        kind = rng.integers(4)
        a, b = rng.choice(4, 2, replace=False)
        if kind == 0:
            c.rx(rng.uniform(0, 2 * np.pi), a)
        elif kind == 1:
            c.ry(rng.uniform(0, 2 * np.pi), a)
        elif kind == 2:
            c.cx(a, b)
        else:
            c.swap(a, b)
    assert_same(c)


def test_sampling_matches_probabilities():
    counts = bell_state().sample(shots=20000, seed=1)
    assert set(counts) == {"00", "11"}
    assert abs(counts["00"] / 20000 - 0.5) < 0.02
