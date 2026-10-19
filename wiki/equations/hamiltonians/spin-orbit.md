# Spin-Orbit Hamiltonian

> Locate spin-linear terms and distinguish general momenta from center-of-mass restrictions. · Status: confirmed
> Up: [Hamiltonians](index.md)

## Summary

`PN_Hamiltonian_SO` stores spin-orbit contributions separately. Each combines
an angular-velocity vector with each body's spin. A term that vanishes for
opposite momenta still belongs to the general Hamiltonian expression.

## Detail

### Contributions and exchange of body labels

The constructor accepts masses, both separation directions, both spin vectors,
both momentum vectors, and `r12`. It constructs `H_SO_1p5PN`, `H_SO_2p5PN`,
and `H_SO_3p5PN`. Each has the form

\[
H_{\rm SO}=\boldsymbol\Omega_1\cdot\mathbf S_1
+\boldsymbol\Omega_2\cdot\mathbf S_2,
\]

where the second angular-velocity expression is built by exchanging body
labels, including `n12U` with `n21U` and `p1U` with `p2U`. The routines
do not replace the independently supplied momenta by opposite vectors.

Claim evidence:
- Claim: PN_Hamiltonian_SO constructs the named 1.5PN, 2.5PN, and 3.5PN contributions by spin contraction and body-label exchange, without imposing p2U = -p1U.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SO.py](../../../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO.__init__`, `f_H_SO_1p5PN`, `f_H_SO_2p5PN`, and `f_H_SO_3p5PN`.
- Corroboration: [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py), derived `p1U` and `p2U`, imposes opposite momenta separately for callers using those shortcuts.

### Leading expression

The leading angular-velocity vector is implemented as

\[
\boldsymbol\Omega_1^{1.5\rm PN}
=\frac{\frac{3m_2}{2m_1}(\mathbf n_{12}\times\mathbf p_1)
-2(\mathbf n_{12}\times\mathbf p_2)}{r_{12}^2}.
\]

The method docstrings attribute the leading and next terms to Damour,
Jaranowski, and Schäfer, and the 3.5PN method to Hartung and Steinhoff.
References in docstrings are bibliographic attributions; this page describes
the implementation rather than independently verifying its derivation.
The broader class docstring is not sufficient to attribute every term to
the leading-order reference.

Claim evidence:
- Claim: The leading Omega expression is as displayed; the per-method docstrings attribute the lower-order terms and the 3.5PN term to different references. These attributions do not independently verify derivation.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SO.py](../../../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO.f_H_SO_1p5PN.f_Omega1`, `f_H_SO_2p5PN`, and `f_H_SO_3p5PN` docstrings.
- Corroboration: none available; the references are bibliographic attributions, and the numerical `trusted_dict` does not corroborate attribution or derivation.

Claim status: contested; contradiction: CONTR-0005.
See [CONTR-0005](../../contradictions.md#contr-0005) for deciding evidence and resolution.

### A term hidden by the center-of-mass restriction

`HS2011_Omega_SO_3p5PN_pt3` contains the contribution

\[
\Delta\boldsymbol\Omega_1=
\frac{(\mathbf n_{12}\cdot\mathbf p_2)(\mathbf p_2\cdot\mathbf p_2)}
{m_1m_2^3r_{12}^2}(\mathbf p_1\times\mathbf p_2).
\]

Its cross product vanishes when \(\mathbf p_2=-\mathbf p_1\), including
the shortcut-based quasicircular calculation. A comparison restricted to that
frame cannot detect omission of this term. The method docstring reports such
an omission in earlier transcriptions; the expression here establishes its
presence in this implementation, not correctness against the cited paper.

Claim evidence:
- Claim: The displayed pt3 contribution is present and vanishes for opposite momenta, so a center-of-mass-only comparison cannot test its presence in the general Hamiltonian.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SO.py](../../../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO.HS2011_Omega_SO_3p5PN_pt3`, expression and explanatory docstring.
- Corroboration: [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py), derived `p1U` and `p2U`, and [eval_p_t_and_p_r.py](../../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`, establish use of that restriction in numerical momentum evaluation; they are not independent scientific derivations.

The registered `trusted_dict` provides references for the named SO expressions.
For the distinction between numerical comparison and scientific evidence, see
[Scientific evidence](../../validation/scientific-evidence.md).

## Sources

- [PN_Hamiltonian_SO.py](../../../nrpypn/PN_Hamiltonian_SO.py) — `PN_Hamiltonian_SO.__init__`, `f_H_SO_1p5PN`, `f_H_SO_2p5PN`, `HS2011_Omega_SO_3p5PN_pt3`, and `f_H_SO_3p5PN`.
- [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py) — derived momentum vectors.
- [eval_p_t_and_p_r.py](../../../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r`.
- [tests/PN_Hamiltonian_SO.py](../../../nrpypn/tests/PN_Hamiltonian_SO.py) — `trusted_dict`.

## See Also

- Parent: [Hamiltonians](index.md)
- Depends on: [Symbols and conventions](../symbols-and-conventions.md)
- See also: [Spin-spin](spin-spin.md)
- Validated by: [Expression regressions](../../validation/expression-regressions.md)
- Contrasts with: [Scientific evidence](../../validation/scientific-evidence.md)
