"""Contextual signal vectors and graded signal fidelity (paper Sec. 3.1).

The paper defines

    sigma(s_r, s*; k) = exp( -||s_r - s*||^2 / (2 delta_k^2) ),
    delta_k = delta_0 (1 + eta k)

but leaves the metric on signal space unspecified.  We instantiate it as a
weighted sum of a temporal term and a semantic-context term:

    ||s_r - s*||^2  =  w_t * (dt / tau_days)^2  +  w_c * (1 - cos(e_r, e*))

where dt is the gap in days between the query's temporal anchor and the
trace's encoding time, and e is a unit-norm embedding of the context
descriptor (participants, location, activity, topic).  If the query carries no
temporal anchor the temporal term is dropped and the semantic term is
renormalised, so an atemporal query is not penalised for lacking a date.

This is an instantiation choice, not a result: other metrics on signal space
are compatible with the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# metric weights
W_TIME = 1.0
W_CTX = 1.0
TAU_DAYS = 45.0  # temporal scale: a 45-day gap contributes ~1 unit of sq. distance


@dataclass
class SignalVector:
    """Contextual signal vector s* recorded at encoding time."""

    t_days: float | None           # encoding time, days since corpus epoch
    ctx_emb: np.ndarray            # unit-norm embedding of the context descriptor
    descriptor: str = ""           # human-readable, for inspection/debugging

    def to_json(self) -> dict:
        return {
            "t_days": self.t_days,
            "ctx_emb": self.ctx_emb.astype(np.float32).tolist(),
            "descriptor": self.descriptor,
        }

    @staticmethod
    def from_json(d: dict) -> "SignalVector":
        return SignalVector(
            t_days=d["t_days"],
            ctx_emb=np.asarray(d["ctx_emb"], dtype=np.float32),
            descriptor=d.get("descriptor", ""),
        )


def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


def delta_k(k: int, delta_0: float, eta: float) -> float:
    """Stage-widening gist window (paper Sec. 3.1).

    Fresh traces are matched narrowly; aged traces are matched loosely, so a
    coarse gist cue suffices to reach them.  This is what keeps old traces
    reachable at all.
    """
    return delta_0 * (1.0 + eta * k)


def sq_distance(q: SignalVector, s: SignalVector) -> float:
    """Squared distance in signal space, with graceful handling of a missing
    temporal anchor on the query side."""
    ctx = 1.0 - float(np.dot(q.ctx_emb, s.ctx_emb))  # in [0, 2]
    ctx = max(0.0, ctx)

    if q.t_days is None or s.t_days is None:
        # No temporal anchor on one side: match on semantic context alone
        # rather than penalising the query for lacking a date.
        return W_CTX * ctx

    dt = abs(q.t_days - s.t_days) / TAU_DAYS
    return W_TIME * dt * dt + W_CTX * ctx


def sigma(
    q: SignalVector, s: SignalVector, k: int, *, delta_0: float, eta: float
) -> float:
    """Graded signal fidelity in (0, 1]."""
    d2 = sq_distance(q, s)
    dk = delta_k(k, delta_0, eta)
    return float(math.exp(-d2 / (2.0 * dk * dk)))
