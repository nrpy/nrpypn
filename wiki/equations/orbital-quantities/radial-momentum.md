# Radial Momentum

> Follow inward drift and distinguish the closed radial expression from the full-Hamiltonian approximation. · Status: confirmed
> Up: [Orbital quantities](index.md)

## Summary

`PN_p_r` builds an instantaneous orbital-plane Hamiltonian and obtains
inward radial drift from energy balance. It provides a default closed
expression and an alternative based on Hamiltonian momentum derivatives.
Comparisons need the same frame, normalization, and expansion order;
pointwise equality of the unexpanded expressions is not promised.

## Detail

### Hamiltonian assembly and inward sign

`f_Htot_xyplane_binary` adds the nonspinning, spin-orbit, spin-spin, and
cubic-spin contributions. With the shortcut vector relations, its replacements
give \(\mathbf n=(1,0,0)\), \(\mathbf p_1=(-P_r,P_t,0)\), and
\(\mathbf p_2=(P_r,-P_t,0)\). It stores the result as
`Htot_xyplane_binary`, denoted \(H\) below. Positive `Pr` is inward for
body 1 at positive x, and `dr_dt` means \(v_{\rm in}=-dr/dt\).
The substitutions target the shortcut `nU` symbols, so arbitrary independent
direction inputs do not automatically receive this orientation.

Claim evidence:
- Claim: With the shortcut vectors, the assembled Hamiltonian uses p1 = (-Pr, Pt, 0), p2 = (Pr, -Pt, 0), and inward-positive Pr; dr_dt denotes minus the separation velocity.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary.make_replacements` and `f_dr_dt`; [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py), derived direction and momentum vectors.
- Corroboration: [quasicirc.py](../../../nrpypn/quasicirc.py), `main`, describes body 1 at positive x and assigns a negative x component to its inward momentum in the initial-data instructions.

### Circular derivative and flux balance

Writing \(H_0(r,P_t)=H|_{P_r=0}\), the drift calculation constructs

\[
D=\left.\frac{\partial H_0}{\partial r}\right|_{P_t}
+\frac{\partial H_0}{\partial P_t}\frac{dp_t(r)}{dr},\qquad
v_{\rm in}=\frac{\dot E_{\rm GW}+\dot M}{D}.
\]

`f_dr_dt` obtains `p_t(r)` and `MOmega(r)` from their expression classes,
then uses `dE_GW_dt_plus_dM_dt` for the numerator. The returned symbolic
drift can still contain `Pt`: the numerical evaluator supplies its value
after evaluating the tangential series. This differentiation holds supplied
masses and spins fixed; it does not include spin evolution along an orbit.
The inward-positive interpretation applies when the evaluated ratio is
positive; the routine does not enforce positivity.

Claim evidence:
- Claim: f_dr_dt constructs the displayed circular derivative and flux ratio at fixed supplied masses and spins, leaving Pt substitution to evaluation and imposing no positivity check.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_dr_dt`, assignments to `dHcirc_dr` and `self.dr_dt`.
- Corroboration: [eval_p_t_and_p_r.py](../../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`, evaluates the tangential expression first and passes `nPt` to radial evaluation; [PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/PN_dE_GW_dt_and_dM_dt.py), `PN_dE_GW_dt_and_dM_dt.f_dE_GW_dt_and_dM_dt`, defines the numerator.

### Alternative based on Hamiltonian derivatives

Setting `use_alternative_approach_for_p_r=True` selects
`f_p_r_alternative__use_fullHam`, which constructs

\[
p_r\approx\frac{v_{\rm in}-A}{B},\qquad
A=\left.\frac{\partial H}{\partial P_r}\right|_{P_r=0},\qquad
B=\left.\frac{\partial^2 H}{\partial P_r^2}\right|_{P_r=0}.
\]

This uses the linear-in-\(P_r\) expansion of the Hamiltonian velocity
relation. “Full” refers to the assembled, finite-order Hamiltonian and the
unexpanded derivative ratio; it is not an exact solution of the nonlinear
Hamilton equation, nor an all-order Hamiltonian. The class docstring
explicitly says this route does not consistently truncate higher-order terms.

Claim evidence:
- Claim: The alternative route evaluates the displayed derivative ratio using the assembled Hamiltonian and a linear radial-momentum approximation, without consistent higher-order truncation or a nonlinear solve.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.__init__`, class docstring, and `f_p_r_alternative__use_fullHam`, including its approximation formula and derivative assignments.
- Corroboration: [tests/PN_p_r_AlternativeTrue.py](../../../nrpypn/tests/PN_p_r_AlternativeTrue.py), `trusted_dict`, contains references for `Htot_xyplane_binary`, `dr_dt`, and `p_r`; these preserve expression values and do not prove the approximation's scientific validity.

### Closed expression and comparison order

The default `f_p_r` computes `p_r_num / p_r_den`. Its numerator starts
with `-self.dr_dt`, followed by in-plane-spin groups multiplying
\(r^{-7/2}\) and \(r^{-4}\). Its denominator starts with
\(-(1+Q)^2/Q\), where \(Q=m_2/m_1\). These leading signs give the
Newtonian reduction \(p_r\approx Qv_{\rm in}/(1+Q)^2\) when the spin
groups and inverse-separation corrections are omitted.

Claim evidence:
- Claim: The closed expression has leading numerator -dr_dt and denominator -(1+Q)^2/Q, giving inward-positive p_r approximately Q dr_dt/(1+Q)^2 at leading order when the additional terms are omitted.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_p_r`, assignments to `p_r_num`, `p_r_den`, and `self.p_r`.
- Corroboration: [PN_Hamiltonian_NS.py](../../../nrpypn/PN_Hamiltonian_NS.py), `PN_Hamiltonian_NS.f_H_Newt__H_NS_1PN__H_NS_2PN`, gives radial kinetic curvature `1/mu`; with unit total mass, `mu = Q/(1+Q)**2`. This is a consistency relation between implementations, not an independent derivation.

The `f_p_r` docstring reports that its in-plane-spin signs and cubic-spin
factors agree with the alternative route through the orders retained.
It uses `(5*q+3)` and `(3*q+5)` in the cubic-spin numerator group and
relates them to the SSS coefficient. This is a repository claim about
matching-order expressions, not a promise that the unexpanded ratios are
identical. To examine that claim, impose the same shortcut frame,
\(m_1+m_2=1\), \(\mathbf S_A=m_A^2\boldsymbol\chi_A\), and tangential
series in both routes, then expand to a common inverse-separation order.
Their separate numerical reference dictionaries do not perform that symbolic
matching or independently establish a scientific derivation.

Claim evidence:
- Claim: The source reports matching-order agreement after its sign and cubic-spin choices; full and closed expressions require common substitutions and expansion order, and the separate trusted dictionaries do not establish that agreement symbolically.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_p_r` docstring and expression, `f_p_r_alternative__use_fullHam`, and module `__main__` branch-specific comparisons.
- Corroboration: [tests/PN_p_r_AlternativeFalse.py](../../../nrpypn/tests/PN_p_r_AlternativeFalse.py) and [tests/PN_p_r_AlternativeTrue.py](../../../nrpypn/tests/PN_p_r_AlternativeTrue.py), `trusted_dict`, store each route's values separately and do not encode an equality test. [PN_Hamiltonian_SSS.py](../../../nrpypn/PN_Hamiltonian_SSS.py), `PN_Hamiltonian_SSS.f_H_SSS_3PN`, supplies the related implemented coefficient, not an independent proof of the matching claim.

References in the docstrings are bibliographic attributions; this page
describes the implementation rather than independently verifying its
derivation or the cited paper's sign convention.

## Sources

- [PN_p_r.py](../../../nrpypn/PN_p_r.py) — `PN_p_r.__init__`, `f_Htot_xyplane_binary`, `f_dr_dt`, `f_p_r_alternative__use_fullHam`, `f_p_r`, and module `__main__` comparisons.
- [NRPyPN_shortcuts.py](../../../nrpypn/NRPyPN_shortcuts.py) — derived vectors and `num_eval`.
- [quasicirc.py](../../../nrpypn/quasicirc.py) — `main`, frame and momentum instructions.
- [eval_p_t_and_p_r.py](../../../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r`.
- [PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/PN_dE_GW_dt_and_dM_dt.py) — `PN_dE_GW_dt_and_dM_dt.f_dE_GW_dt_and_dM_dt`.
- [PN_Hamiltonian_NS.py](../../../nrpypn/PN_Hamiltonian_NS.py) — `PN_Hamiltonian_NS.f_H_Newt__H_NS_1PN__H_NS_2PN`.
- [PN_Hamiltonian_SSS.py](../../../nrpypn/PN_Hamiltonian_SSS.py) — `PN_Hamiltonian_SSS.f_H_SSS_3PN`.
- [tests/PN_p_r_AlternativeFalse.py](../../../nrpypn/tests/PN_p_r_AlternativeFalse.py) — `trusted_dict`.
- [tests/PN_p_r_AlternativeTrue.py](../../../nrpypn/tests/PN_p_r_AlternativeTrue.py) — `trusted_dict`.

## See Also

- Parent: [Orbital quantities](index.md)
- Depends on: [Symbols and conventions](../symbols-and-conventions.md)
- Depends on: [Tangential momentum and frequency](tangential-momentum-and-frequency.md)
- Depends on: [Fluxes](fluxes.md)
- See also: [Spin-spin](../hamiltonians/spin-spin.md)
- Example: [Numerical evaluation](../../usage/numerical-evaluation.md)
- Contrasts with: [Scientific evidence](../../validation/scientific-evidence.md)
- Validated by: [Expression regressions](../../validation/expression-regressions.md)
