"""A lightweight statevector quantum circuit simulator.

The state of n qubits is stored as a tensor of shape (2,)*n, so applying a
k-qubit gate is a single np.tensordot over the target axes. Qubit 0 is the
most significant (leftmost) bit of a basis-state label like |q0 q1 q2>.
"""
import numpy as np
from . import gates as g


class Circuit:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.ops = []  # list of (name, matrix, wires)

    # ------------------------------------------------------------------ build
    def apply(self, name, matrix, *wires):
        matrix = np.asarray(matrix, dtype=complex)
        k = len(wires)
        if matrix.shape != (2 ** k, 2 ** k):
            raise ValueError(f"{name}: matrix shape {matrix.shape} does not match {k} wire(s)")
        if len(set(wires)) != k or any(w < 0 or w >= self.n for w in wires):
            raise ValueError(f"{name}: invalid wires {wires}")
        self.ops.append((name, matrix, tuple(wires)))
        return self

    def h(self, q): return self.apply("H", g.H, q)
    def x(self, q): return self.apply("X", g.X, q)
    def y(self, q): return self.apply("Y", g.Y, q)
    def z(self, q): return self.apply("Z", g.Z, q)
    def s(self, q): return self.apply("S", g.S, q)
    def t(self, q): return self.apply("T", g.T, q)
    def rx(self, theta, q): return self.apply(f"RX({theta:.2f})", g.RX(theta), q)
    def ry(self, theta, q): return self.apply(f"RY({theta:.2f})", g.RY(theta), q)
    def rz(self, theta, q): return self.apply(f"RZ({theta:.2f})", g.RZ(theta), q)
    def cx(self, c, t): return self.apply("CX", g.CNOT, c, t)
    def cz(self, c, t): return self.apply("CZ", g.CZ, c, t)
    def swap(self, a, b): return self.apply("SWAP", g.SWAP, a, b)

    def mcz(self, *wires):
        """Multi-controlled Z on all given wires (flips the phase of |11...1>)."""
        k = len(wires)
        m = np.eye(2 ** k, dtype=complex)
        m[-1, -1] = -1
        return self.apply("MCZ", m, *wires)

    # --------------------------------------------------------------- simulate
    def statevector(self, initial=None):
        if initial is None:
            psi = np.zeros(2 ** self.n, dtype=complex)
            psi[0] = 1.0
        else:
            psi = np.asarray(initial, dtype=complex).copy()
        psi = psi.reshape((2,) * self.n)
        for _, m, wires in self.ops:
            psi = self._apply_matrix(psi, m, wires)
        return psi.reshape(-1)

    def _apply_matrix(self, psi, m, wires):
        k = len(wires)
        m = m.reshape((2,) * (2 * k))
        # contract gate's input axes with the state's target axes
        psi = np.tensordot(m, psi, axes=(list(range(k, 2 * k)), list(wires)))
        # tensordot puts the gate's output axes first; move them back
        return np.moveaxis(psi, list(range(k)), list(wires))

    def probabilities(self):
        return np.abs(self.statevector()) ** 2

    def sample(self, shots=1024, seed=None):
        rng = np.random.default_rng(seed)
        p = self.probabilities()
        p = p / p.sum()
        outcomes = rng.choice(len(p), size=shots, p=p)
        counts = {}
        for o in outcomes:
            key = format(o, f"0{self.n}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------- draw
    def draw(self):
        """Return a simple text diagram of the circuit."""
        lines = [f"q{i}: " for i in range(self.n)]
        for name, _, wires in self.ops:
            width = max(len(name), 3) + 2
            for q in range(self.n):
                if q in wires:
                    if len(wires) == 1:
                        cell = f"[{name}]"
                    elif name in ("CX", "CZ") and q == wires[0]:
                        cell = "●"
                    elif name == "CX":
                        cell = "⊕"
                    elif name == "CZ":
                        cell = "●"
                    else:
                        cell = f"[{name}]"
                elif len(wires) > 1 and min(wires) < q < max(wires):
                    cell = "│"
                else:
                    cell = ""
                lines[q] += cell.center(width, "─")
        return "\n".join(l + "─" for l in lines)

    def __str__(self):
        return self.draw()
