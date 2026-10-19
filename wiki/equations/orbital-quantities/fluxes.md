# Energy Flux and Mass Change

> Identify the implemented frequency expansion and the flux combination used for radial drift. · Status: confirmed
> Up: [Orbital quantities](index.md)

## Summary

`PN_dE_GW_dt_and_dM_dt` constructs gravitational-wave energy flux, mass
change, and their sum as separate expressions. The radial calculation uses
the sum. The defining expressions specify which frequency powers and spin
terms are included; a general “highest known order” description does not
define that content.

## Detail

### Frequency variable and expansion

The input `mOmega` is used as the dimensionless frequency \(M\Omega\),
with \(x=(M\Omega)^{2/3}\) and
\(\nu=m_1m_2/(m_1+m_2)^2\). The energy-flux expression is

\[
\dot E_{\rm GW}=\frac{32}{5}\nu^2x^5
\left[1+\sum_{k=2}^{8}b_kx^{k/2}\right].
\]

The implemented `b[8]` contains only terms linear in the spin projections
`s_l` and `delta*sigma_l`. Its presence is not evidence that every physical
effect through the relative \(x^4\) power is included. `b[6]` includes
`sp.log(16*x)` and `gamma_EulerMascheroni`.

Claim evidence:
- Claim: The energy flux uses the displayed x expansion through b[8], with only spin-linear terms in b[8] and logarithmic and Euler-constant terms in b[6]; this does not establish complete physical content at that power.
- Role: descriptive behavior
- Deciding authority: [PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/PN_dE_GW_dt_and_dM_dt.py), `PN_dE_GW_dt_and_dM_dt.f_dE_GW_dt_and_dM_dt.f_compute_quantities`, assignments to `x`, `b[6]`, `b[8]`, and `b_sum`.
- Corroboration: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_dr_dt`, passes `PN_MOmega.MOmega` as this frequency input. The numerical reference dictionary corroborates expression regression only, not completeness of scientific terms.

### Spin projections and fixed orbital normal

`dE_GW_dt_OBKPSS2015_consts` defines
\(\mathbf l=(0,0,1)\),
\(\boldsymbol\chi_s=(\boldsymbol\chi_1+\boldsymbol\chi_2)/2\),
\(\boldsymbol\chi_a=(\boldsymbol\chi_1-\boldsymbol\chi_2)/2\), and
\(\delta=(m_1-m_2)/(m_1+m_2)\), reconstructing
\(\boldsymbol\chi_A=\mathbf S_A/m_A^2\). It also defines

\[
\mathbf\Sigma=\frac{m_1+m_2}{m_2}\mathbf S_2
-\frac{m_1+m_2}{m_1}\mathbf S_1,\qquad
s_l=\frac{(\mathbf S_1+\mathbf S_2)\cdot\mathbf l}{(m_1+m_2)^2},\qquad
\sigma_l=\frac{\mathbf\Sigma\cdot\mathbf l}{(m_1+m_2)^2}.
\]

Its `_n12U` parameter is unused. The orbital normal is set internally;
passing a differently oriented separation vector does not rotate this flux
formula. Spin dot products in `b[4]` also retain dependence beyond the
z projections.

Claim evidence:
- Claim: The flux constants use the displayed spin definitions and a fixed z-directed orbital normal, ignore the supplied separation direction, and retain full spin dot products in b[4].
- Role: descriptive behavior
- Deciding authority: [PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/PN_dE_GW_dt_and_dM_dt.py), `PN_dE_GW_dt_and_dM_dt.dE_GW_dt_OBKPSS2015_consts` and `f_dE_GW_dt_and_dM_dt.f_compute_quantities`, `b[4]`.
- Corroboration: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary`, establishes the compatible instantaneous xy-plane frame for the shortcut-based radial path. It is a caller restriction, not a separate derivation of the flux formula.

### Mass change and the summed flux

Writing `b_5_Mdot` as \(b_5^{(M)}\), the implementation defines

\[
\dot M=\frac{32}{5}\nu^2x^5b_5^{(M)}x^{5/2}.
\]

The summed expression adds `b_5_Mdot` to `b[5]`, so its construction gives
`dE_GW_dt_plus_dM_dt = dE_GW_dt + dM_dt`. `b_5_Mdot` has an overall
negative factor and spin-dependent terms; it is not a positive magnitude
by definition and vanishes for zero spins. `PN_p_r.f_dr_dt` uses this
sum directly as the numerator of its inward drift ratio, without an
additional minus sign.

Claim evidence:
- Claim: The mass-change expression is as displayed, the summed flux equals the energy expression plus that mass change by construction, and the radial drift numerator uses that sum directly; mass change is not defined as an absolute value.
- Role: descriptive behavior
- Deciding authority: [PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/PN_dE_GW_dt_and_dM_dt.py), `PN_dE_GW_dt_and_dM_dt.f_dE_GW_dt_and_dM_dt.f_compute_quantities`, `b_5_Mdot`, mass-only return, and `b[5]` addition; [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_dr_dt`.
- Corroboration: [tests/PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/tests/PN_dE_GW_dt_and_dM_dt.py), `trusted_dict`, stores the separate and summed expressions' numerical references; it does not establish a general physical sign or a scientific derivation.

### Bibliographic scope

The class and method docstrings attribute the flux formulas and a `b[7]`
correction to external references. References in docstrings are bibliographic
attributions; this page describes the implementation rather than independently
verifying its derivation. The coefficient assignments, not the module's
“highest known” wording, decide which terms the code constructs.

Claim evidence:
- Claim: The docstrings provide bibliographic attributions and report a b[7] correction; the implemented coefficient assignments determine the expression content without establishing a highest-known-order scientific guarantee.
- Role: descriptive behavior
- Deciding authority: [PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/PN_dE_GW_dt_and_dM_dt.py), module docstring, `PN_dE_GW_dt_and_dM_dt` class docstring, and `f_dE_GW_dt_and_dM_dt` docstring and assignments.
- Corroboration: none available for independent verification of the cited derivations or highest-known-order wording; the registered numerical references only preserve expression values.

## Sources

- [PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/PN_dE_GW_dt_and_dM_dt.py) — `PN_dE_GW_dt_and_dM_dt.__init__`, `dE_GW_dt_OBKPSS2015_consts`, `f_dE_GW_dt_and_dM_dt`, and its nested `f_compute_quantities`.
- [PN_p_r.py](../../../nrpypn/PN_p_r.py) — `PN_p_r.f_Htot_xyplane_binary` and `f_dr_dt`.
- [tests/PN_dE_GW_dt_and_dM_dt.py](../../../nrpypn/tests/PN_dE_GW_dt_and_dM_dt.py) — `trusted_dict`.

## See Also

- Parent: [Orbital quantities](index.md)
- Depends on: [Symbols and conventions](../symbols-and-conventions.md)
- See also: [Tangential momentum and frequency](tangential-momentum-and-frequency.md)
- See also: [Radial momentum](radial-momentum.md)
- Contrasts with: [Scientific evidence](../../validation/scientific-evidence.md)
- Validated by: [Expression regressions](../../validation/expression-regressions.md)
