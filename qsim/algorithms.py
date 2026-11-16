"""Example circuits built with qsim."""
import numpy as np
from .circuit import Circuit


def bell_state():
    return Circuit(2).h(0).cx(0, 1)


def ghz_state(n):
    c = Circuit(n).h(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c


def teleportation(theta, phi):
    """Teleport RY(theta)RZ(phi)|0> from q0 to q2 using deferred measurement.

    Classically controlled corrections are replaced by controlled gates
    (principle of deferred measurement), so the whole protocol is unitary.
    """
    c = Circuit(3)
    c.ry(theta, 0).rz(phi, 0)          # state to send
    c.h(1).cx(1, 2)                    # shared Bell pair between q1 and q2
    c.cx(0, 1).h(0)                    # Alice's Bell-basis rotation
    c.cx(1, 2).cz(0, 2)                # Bob's corrections
    return c


def grover(n, marked):
    """Grover search on n qubits for the basis state `marked` (a bit string)."""
    c = Circuit(n)
    wires = list(range(n))
    for q in wires:
        c.h(q)
    iterations = int(np.floor(np.pi / 4 * np.sqrt(2 ** n)))
    for _ in range(iterations):
        # Oracle: phase flip on |marked>
        for q, bit in enumerate(marked):
            if bit == "0":
                c.x(q)
        c.mcz(*wires)
        for q, bit in enumerate(marked):
            if bit == "0":
                c.x(q)
        # Diffusion operator: 2|s><s| - I
        for q in wires:
            c.h(q)
            c.x(q)
        c.mcz(*wires)
        for q in wires:
            c.x(q)
            c.h(q)
    return c
