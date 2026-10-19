# Nonspinning Hamiltonian

> Interpret the nonspinning Hamiltonian expressions and their momentum normalization. · Status: confirmed
> Up: [Hamiltonians](index.md)

## Summary

`PN_Hamiltonian_NS` constructs separate Newtonian and post-Newtonian
contributions. Its input `PU` is rescaled internally by the reduced mass.
Read its separation and momentum together with the numerical normalization
before using the expressions outside the quasicircular driver.

## Detail

### Inputs and stored contributions

The constructor `PN_Hamiltonian_NS(m1, m2, PU, nU, q)` stores `H_Newt`,
`H_NS_1PN`, `H_NS_2PN`, and `H_NS_3PN` as separate expressions. It does not
store their sum. Here `q` is the separation argument, not `m2/m1`.

Claim evidence:
- Claim: PN_Hamiltonian_NS constructs separate Newtonian, 1PN, 2PN, and 3PN expressions with q as separation and no total-Hamiltonian attribute.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_NS.py](../../../nrpypn/PN_Hamiltonian_NS.py), `PN_Hamiltonian_NS.__init__`, `f_H_Newt__H_NS_1PN__H_NS_2PN`, and `f_H_NS_3PN`.
- Corroboration: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary`, passes `r` for this argument and explicitly adds the stored terms.

### Reduced momentum

Both expression-building methods define

\[
\mu=\frac{m_1m_2}{m_1+m_2},\qquad
\eta=\frac{m_1m_2}{(m_1+m_2)^2},\qquad
\mathbf p=\frac{\mathbf P}{\mu}.
\]

Writing \(u=\mathbf p\cdot\mathbf p\) and
\(w=\mathbf n\cdot\mathbf p\), the expressions begin with

\[
H_{\rm Newt}=\mu\left(\frac{u}{2}-\frac{1}{q}\right),\qquad
H_{\rm NS,1PN}=\mu\left[
\frac{3\eta-1}{8}u^2
-\frac{(3+\eta)u+\eta w^2}{2q}
+\frac{1}{2q^2}\right].
\]

These are the expressions as implemented. Passing an already reduced momentum
as `PU` would apply the division by `mu` again. The numerical path sets
\(m_1+m_2=1\); the displayed potential should not be generalized to an
arbitrary dimensional separation without establishing the corresponding units.

Claim evidence:
- Claim: The nonspinning methods divide PU by mu and use the displayed Newtonian and 1PN expressions; num_eval supplies a unit-total-mass numerical normalization.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_NS.py](../../../nrpypn/PN_Hamiltonian_NS.py), `PN_Hamiltonian_NS.f_H_Newt__H_NS_1PN__H_NS_2PN` and `f_H_NS_3PN`; [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py), `num_eval`.
- Corroboration: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary`, passes body momentum `p1U` directly, without a prior reduced-mass division. This corroborates the calling convention, not a scientific derivation or a dimensional extension.

### Direction wording in the shared calculation

The class parameter docstring says `nU` points from body 1 to body 2.
The shared shortcuts instead set `n12U = nU`, and the spin Hamiltonian
parameter definition gives `n12U = (x1-x2)/r12`, from body 2 to body 1.
The radial caller passes the shared `nU` to the nonspinning class. Use that
caller convention for the assembled calculation; the differing wording alone
does not establish an error in the nonspinning Hamiltonian or license a sign
change in another term.

Claim evidence:
- Claim: The nonspinning class's direction description differs from the shared radial caller's nU = n12U convention; the discrepancy alone establishes no nonspinning equation sign error.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_NS.py](../../../nrpypn/PN_Hamiltonian_NS.py), `PN_Hamiltonian_NS` parameter docstring; [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py), derived `n12U`; [PN_Hamiltonian_SO.py](../../../nrpypn/PN_Hamiltonian_SO.py), `n12U` definition.
- Corroboration: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary`, passes the shared nU and supplies the instantaneous orientation.

Claim status: contested; contradiction: CONTR-0003.
See [CONTR-0003](../../contradictions.md#contr-0003) for the direction comparison and resolution test.

### Higher-order expressions and evidence

`f_H_NS_3PN` includes terms proportional to `sp.pi**2` and inverse separation
powers through `q**(-4)`. The checked-in `trusted_dict` stores numerical
reference values for the named contributions. Those values support expression
regression; they do not establish the derivation of the Hamiltonian.

Claim evidence:
- Claim: f_H_NS_3PN includes pi-squared coefficients and inverse separation through q^(-4); its stored numerical references are expression-regression evidence only.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_NS.py](../../../nrpypn/PN_Hamiltonian_NS.py), `PN_Hamiltonian_NS.f_H_NS_3PN` and module `__main__` expression-comparison block.
- Corroboration: [tests/PN_Hamiltonian_NS.py](../../../nrpypn/tests/PN_Hamiltonian_NS.py), `trusted_dict`, supplies references keyed by the Hamiltonian attributes; it supplies no derivation.

## Sources

- [PN_Hamiltonian_NS.py](../../../nrpypn/PN_Hamiltonian_NS.py) — `PN_Hamiltonian_NS`, both expression-building methods, and module `__main__` comparison block.
- [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py) — `num_eval`.
- [PN_Hamiltonian_SO.py](../../../nrpypn/PN_Hamiltonian_SO.py) — `n12U` parameter definition.
- [PN_p_r.py](../../../nrpypn/PN_p_r.py) — `PN_p_r.f_Htot_xyplane_binary`.
- [tests/PN_Hamiltonian_NS.py](../../../nrpypn/tests/PN_Hamiltonian_NS.py) — `trusted_dict`.

## See Also

- Parent: [Hamiltonians](index.md)
- Depends on: [Symbols and conventions](../symbols-and-conventions.md)
- See also: [Spin-orbit](spin-orbit.md)
- Validated by: [Expression regressions](../../validation/expression-regressions.md)
