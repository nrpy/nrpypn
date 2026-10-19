# Tangential Momentum and Orbital Frequency

> Read the separation series and identify the coefficients used by the radial calculation. · Status: confirmed
> Up: [Orbital quantities](index.md)

## Summary

`PN_p_t` and `PN_MOmega` construct separate quasicircular separation series
using mass ratio and dimensionless spin components. Their coefficient
attributes share names, but belong to different expressions. The source
documents sign and cubic-spin choices that matter when comparing references.

## Detail

### Series and coefficient names

Writing \(Q=m_2/m_1\), the implemented series are

\[
p_t=\frac{Q}{(1+Q)^2\sqrt r}
\left(1+\sum_{k=2}^{7}\frac{a_k^{(t)}}{r^{k/2}}\right),\qquad
M\Omega=\frac{1}{r^{3/2}}
\left(1+\sum_{k=2}^{7}\frac{a_k^{(\Omega)}}{r^{k/2}}\right).
\]

The constructors take `(m1, m2, chi1U, chi2U, r)` and store `.p_t` or
`.MOmega`, together with their own `a_2` through `a_7`. For example,
\(a_2^{(t)}=2\), whereas
\(a_2^{(\Omega)}=-(3Q^2+5Q+3)/(2(1+Q)^2)\).
The defining module docstrings describe the series as including terms
through 3.5PN. Their leading factors use the unit-total-mass numerical
normalization described in [Symbols and conventions](../symbols-and-conventions.md).

Claim evidence:
- Claim: The displayed series, distinct coefficient attributes, and leading a_2 values are implemented by PN_p_t and PN_MOmega, whose module docstrings describe terms through 3.5PN; numerical evaluation normalizes total mass to unity.
- Role: descriptive behavior
- Deciding authority: [PN_p_t.py](../../../nrpypn/PN_p_t.py), module docstring, `PN_p_t.__init__`, `p_t__a_2_thru_a_4`, and `f_p_t`; [PN_MOmega.py](../../../nrpypn/PN_MOmega.py), module docstring, `PN_MOmega.__init__`, `MOmega__a_2_thru_a_4`, and `f_MOmega`; [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py), `num_eval`.
- Corroboration: [tests/PN_p_t.py](../../../nrpypn/tests/PN_p_t.py) and [tests/PN_MOmega.py](../../../nrpypn/tests/PN_MOmega.py), `trusted_dict`, contain references for the coefficients and final expressions. They support numerical regression, not independent scientific PN classification.

### Tangential sign choice

`PN_p_t.p_t__a_5_thru_a_6` defaults `FixSignError=True`, giving
`SignFix = -1` for the first body's linear-spin term in `a_5`:

\[
-\frac{(13Q^3+60Q^2+116Q+72)\chi_{1z}}{16(1+Q)^4}.
\]

`f_p_t` calls this method with its default. The class constructor has no
`FixSignError` parameter. A later direct call to the coefficient method
changes coefficient attributes, but does not itself rebuild the stored
`.p_t` expression. The method docstring attributes the sign choice to a
comparison between bibliographic references; that attribution is not an
independent derivation of the sign.

Claim evidence:
- Claim: The standard constructor uses the displayed negative first-body a_5 term through the default FixSignError=True; directly recomputing coefficients does not automatically reconstruct p_t.
- Role: descriptive behavior
- Deciding authority: [PN_p_t.py](../../../nrpypn/PN_p_t.py), `PN_p_t.p_t__a_5_thru_a_6`, `__init__`, and `f_p_t`.
- Corroboration: none available for independent scientific verification of the sign; the registered coefficient references preserve the implemented values and do not independently derive them.

### Cubic-spin coefficients and radial use

`p_t__a_7` and `MOmega__a_7` contain cubic-spin terms and explicitly
relate their coefficients to the `-9/4` block in `PN_Hamiltonian_SSS`.
These are separately written formulas; neither class derives its expression
by differentiating that Hamiltonian at construction time. Their docstrings
report differences from cited printed expressions. References in docstrings
are bibliographic attributions; this page describes the implementation
rather than independently verifying its derivation.

Claim evidence:
- Claim: The implemented a_7 formulas include cubic-spin terms attributed to the SSS coefficient, but the constructors assemble separately written coefficients rather than deriving them from the Hamiltonian.
- Role: descriptive behavior
- Deciding authority: [PN_p_t.py](../../../nrpypn/PN_p_t.py), `PN_p_t.p_t__a_7` and `f_p_t`; [PN_MOmega.py](../../../nrpypn/PN_MOmega.py), `PN_MOmega.MOmega__a_7` and `f_MOmega`.
- Corroboration: [PN_Hamiltonian_SSS.py](../../../nrpypn/PN_Hamiltonian_SSS.py), `PN_Hamiltonian_SSS.f_H_SSS_3PN.f_H_SSS_3PN_pt`, implements the cited `-9/4` factor. This is a related implementation, not independent evidence for the derivation of a_7.

In `PN_p_r.f_dr_dt`, the tangential series supplies \(dp_t/dr\) in the
circular-Hamiltonian derivative, and `.MOmega` supplies the flux input.
The numerical driver evaluates `.p_t` first and passes its numerical value
as `nPt` when evaluating `.p_r`.

Claim evidence:
- Claim: The radial drift calculation uses the tangential series derivative and the frequency series; the numerical driver evaluates p_t before substituting it into p_r.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_dr_dt`; [eval_p_t_and_p_r.py](../../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`.
- Corroboration: [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py), `num_eval`, substitutes `nPt` for `Pt` when the supplied value is truthy.

## Sources

- [PN_p_t.py](../../../nrpypn/PN_p_t.py) — `PN_p_t.__init__`, `p_t__a_2_thru_a_4`, `p_t__a_5_thru_a_6`, `p_t__a_7`, and `f_p_t`.
- [PN_MOmega.py](../../../nrpypn/PN_MOmega.py) — `PN_MOmega.__init__`, `MOmega__a_2_thru_a_4`, `MOmega__a_7`, and `f_MOmega`.
- [PN_Hamiltonian_SSS.py](../../../nrpypn/PN_Hamiltonian_SSS.py) — `PN_Hamiltonian_SSS.f_H_SSS_3PN`.
- [PN_p_r.py](../../../nrpypn/PN_p_r.py) — `PN_p_r.f_dr_dt`.
- [eval_p_t_and_p_r.py](../../../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r`.
- [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py) — `num_eval`.
- [tests/PN_p_t.py](../../../nrpypn/tests/PN_p_t.py) — `trusted_dict`.
- [tests/PN_MOmega.py](../../../nrpypn/tests/PN_MOmega.py) — `trusted_dict`.

## See Also

- Parent: [Orbital quantities](index.md)
- Depends on: [Symbols and conventions](../symbols-and-conventions.md)
- See also: [Cubic in spin](../hamiltonians/cubic-in-spin.md)
- See also: [Radial momentum](radial-momentum.md)
- Example: [Numerical evaluation](../../usage/numerical-evaluation.md)
- Validated by: [Expression regressions](../../validation/expression-regressions.md)
