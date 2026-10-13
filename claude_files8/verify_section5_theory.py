"""Executable sanity audit for Section 5 / Appendix A.
Part 1 needs no checkpoint (pure math). Part 2 is optional: pass an .npz with
arrays  logits (N,8)  and  z_true (N,8)  for the C35-C38 test wafers (and cal scores)
to check the Theorem-1 implication on your real model outputs."""
import itertools, math, sys
import numpy as np
from math import comb, log, exp

sig = lambda x: 1/(1+np.exp(-x))
V = [z for z in itertools.product([0,1], repeat=8)
     if not (z[0] and z[1]) and not (z[2] and z[3]) and not ((z[5] or z[7]) and sum(z) != 1)]
V = np.array(V, float); K = V[V.sum(1) < 4]; Q = V[V.sum(1) == 4]
print(f"|V|={len(V)}  |V_known|={len(K)}  |depth-4|={len(Q)}  (expect 38, 34, 4)")
assert len(V) == 38 and len(K) == 34 and len(Q) == 4

def slib(l, cfgs=K):
    lp, l1 = -np.logaddexp(0, -l), -np.logaddexp(0, l)           # log sig(l), log(1-sig(l))
    lm = lp @ cfgs.T + l1 @ (1-cfgs).T
    m = lm.max(1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(lm-m).sum(1)))

# ---- Part 1: Monte-Carlo check of Theorem 1 (i)-(iii) and Corollary 1 --------------
rng = np.random.default_rng(0); viol = 0; N = 200000
for zq in Q:
    A = np.where(zq == 1)[0]; n = len(A); M = n-3
    tau = rng.uniform(-1.5, 6, N)                                  # per-sample margin
    L = rng.normal(0, 3, (N, 8))
    L[:, A] = tau[:, None] + rng.exponential(1.0, (N, len(A)))      # all active logits >= tau
    s = slib(L)
    mass = np.exp(-s); eps = 1/(1+np.exp(tau))
    b_i  = comb(n, M)*eps**M
    b_ii = M*np.logaddexp(0, tau) - log(comb(n, M))
    b_iii = -np.log1p(-sig(tau)**n)
    viol += (mass > b_i*(1+1e-9)).sum() + (s < b_ii-1e-9).sum() + (s < b_iii-1e-9).sum()
print("Theorem 1 violations over 4 x 200k random logit draws:", int(viol))
assert viol == 0

qhat = 0.0101
tau_star = math.log((s0 := (1-exp(-qhat))**0.25)/(1-s0))
print(f"Corollary 1 exact threshold tau*(qhat=0.0101) = {tau_star:.4f}  (NOT -0.77; -0.77 < tau*)")
b = lambda t: -log(1-(1/(1+exp(-t)))**4)
print(f"bound at tau=-0.77: {b(-0.77):.5f} (< 0.0101);  at tau=-0.769: {b(-0.769):.5f};  at tau*: {b(tau_star):.5f}")

# trivial-rule dominance: >=4 positive logits  =>  S_lib >= -log(1-2^-4) = 0.0645 > qhat
L = rng.normal(0, 3, (400000, 8)); pos = (L > 0).sum(1) >= 4
print("min S_lib over samples with >=4 positive logits:", slib(L[pos]).min(), ">= bound", -log(1-1/16))

# ---- Part 2 (optional, real model outputs) ----------------------------------------
if len(sys.argv) > 1:
    d = np.load(sys.argv[1]); l, zt = d["logits"], d["z_true"]
    s = slib(l); mins = np.where(zt == 1, l, np.inf).min(1)
    flagged = mins > tau_star
    print("OOD wafers with min active logit > tau*:", int(flagged.sum()),
          "| of those rejected (S_lib>qhat):", int((s[flagged] > qhat).sum()), "(must be equal)")
    assert (s[flagged] > qhat).all()
    print("OOD rejected:", int((s > qhat).sum()), "/", len(s))
