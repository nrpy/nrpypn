# Scientific Corrections

> Trace documented spin-term corrections and the limits of earlier comparisons. · Status: confirmed
> Up: [Validation](index.md)

## Summary

The README identifies inherited errors in cubic-in-spin, spin-spin, spin-orbit,
and closed-form radial-momentum expressions. The affected docstrings explain
why comparisons missed them and which quantities change. Their mathematical
justifications and the numerical references updated afterward provide different
kinds of evidence.

## Detail

### Cubic-In-Spin Coefficient And Dependent Expansions

`PN_Hamiltonian_SSS` documents and implements coefficient `-9/4` for the
block proportional to `m1/m2`, where the inherited transcription used
`-3/2`. Its docstring attributes the grouping to Levi and Steinhoff,
Eq. 3.12. `PN_p_t.p_t__a_7` and `PN_MOmega.MOmega__a_7` document the
resulting cubic-in-spin changes, and `PN_p_r.f_p_r` records corresponding
changes in the closed-form radial momentum. Agreement between transcriptions
sharing the inherited coefficient could not detect that error. References in
docstrings are bibliographic attributions; this page describes the
implementation rather than independently verifying its derivation.

Claim evidence:
- Claim: The repository documents the cubic-in-spin block's correction from `-3/2` to `-9/4` and propagates it into the tangential-momentum, orbital-frequency, and closed-form radial-momentum expressions.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SSS.py](../../nrpypn/PN_Hamiltonian_SSS.py), `PN_Hamiltonian_SSS` docstring and `f_H_SSS_3PN`
- Corroboration: [PN_p_t.py](../../nrpypn/PN_p_t.py), `PN_p_t.p_t__a_7`; [PN_MOmega.py](../../nrpypn/PN_MOmega.py), `PN_MOmega.MOmega__a_7`; [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_p_r`

The module also reports a reciprocal mass-prefactor warning for Hergt and
Schaefer Eq. 2.13, separate from that grouping correction. Follow the
[Cubic-in-Spin Hamiltonian](../equations/hamiltonians/cubic-in-spin.md#coefficient-and-body-exchange)
for the printed and reported-needed prefactors and the evidence limit. The
warning is a source-docstring report, not an independently verified paper correction.

Claim evidence:
- Claim: The cubic-spin module reports an additional reciprocal mass-prefactor warning, distinct from its -3/2 to -9/4 grouping correction, without independent publication verification in this KB.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SSS.py](../../nrpypn/PN_Hamiltonian_SSS.py), `PN_Hamiltonian_SSS` class docstring, Hergt and Schaefer Eq. 2.13 discussion
- Corroboration: none available from independently verified publication evidence; the linked equation page organizes the same source and is not independent authority.

The cubic-in-spin module explains that its leading term is 3.5PN for rapidly
rotating bodies despite the historical `H_SSS_3PN` identifier. Use that
explanation when assigning physical order; the identifier alone does not
decide the order.

Claim evidence:
- Claim: `H_SSS_3PN` retains a historical name for a leading cubic-in-spin contribution that the module identifies as 3.5PN for rapidly rotating bodies.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SSS.py](../../nrpypn/PN_Hamiltonian_SSS.py), module and `PN_Hamiltonian_SSS` docstrings
- Corroboration: [agent guide](../../.agents/README.md), `## Scientific Standards`, instruction on historical PN labels

### Spin-Spin Momentum Substitution

`f_H_SS_S1S2_3PN` uses `S1U` dotted with `p2U` in the
`3/(2*m2**2*r12**3)` block, replacing an inherited `S1U` dotted with
`p1U`. The docstring attributes this form to Steinhoff, Hergt, and Schäfer,
Eq. 2.11. It records that both compared transcriptions shared the slip. The
term's explicit `p2U` dotted with `n12U` factor removes it on circular
orbits, including its separation and tangential-momentum derivatives, while
its radial-momentum derivatives need not vanish. The documented effect is on
the full-Hamiltonian radial-momentum route, with tangential momentum and
orbital frequency unchanged.

Claim evidence:
- Claim: The repository documents an `S1U.p1U` to `S1U.p2U` correction whose circular-orbit value vanishes but whose radial-momentum derivatives affect the full-Hamiltonian radial-momentum route.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SS.py](../../nrpypn/PN_Hamiltonian_SS.py), `PN_Hamiltonian_SS.f_H_SS_S1S2_3PN`, docstring and `3/(2*m2**2*r12**3)` block
- Corroboration: [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary` and `f_p_r_alternative__use_fullHam`; [README.md](../../README.md), `## Validation`

### Spin-Orbit Term Hidden In The Centre-Of-Mass Frame

`HS2011_Omega_SO_3p5PN_pt3` contains the bracket term
`+(n12U.p2U)*(p2U.p2U)/(m1*m2**3)` multiplying
`cross(p1U,p2U)/r12**2`. Its docstring attributes the term to Hartung and
Steinhoff and explains that the earlier comparison imposed `p2U = -p1U`.
The cross product vanishes under that restriction, so the comparison could
not reveal the omission. The documented centre-of-mass momenta and orbital
frequency remain unchanged by this correction, although the general
Hamiltonian expression changes.

Claim evidence:
- Claim: The restored spin-orbit term changes the general expression but vanishes when `p2U = -p1U`, explaining why the documented restricted comparison missed it.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SO.py](../../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO.HS2011_Omega_SO_3p5PN_pt3`, docstring and final bracket term
- Corroboration: [agent guide](../../.agents/README.md), `## Scientific Standards`, requirement to test independent momenta

### Closed-Form Radial-Momentum Signs

`PN_p_r.f_p_r` documents changed signs for the in-plane-spin groups at
`r**(-7/2)` and `r**(-4)`, using the module's convention
`dr_dt = -dr/dt`. Its stated mathematical requirement is agreement with
Hamilton's equation and the full-Hamiltonian route through the retained
orders. The docstring describes a possible convention mismatch with the
printed formula and explicitly leaves that publication-convention explanation
unverified. It must not be promoted into a confirmed diagnosis of the paper.

Claim evidence:
- Claim: The closed-form in-plane-spin signs follow the implementation's radial-velocity convention; the docstring's proposed explanation in terms of publication conventions remains unverified.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_p_r`, `Departures from the transcription`, and `f_dr_dt`
- Corroboration: [agent guide](../../.agents/README.md), `## Scientific Standards`, radial-velocity convention and matching-PN-order comparison requirements

The same docstring says the tutorial's printed Table V comparison labels were
decided using the full-Hamiltonian route and did not assert the closed form.
A displayed comparison label therefore cannot be used as a closed-form
regression assertion. The evaluator's `case_dict` comments separately record
regenerated spinning reference values after the coefficient and sign changes.

Claim evidence:
- Claim: The repository describes the tutorial's Table V labels as non-asserting and based on the full-Hamiltonian route; the evaluator's updated stored values are subsequent regression references.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_p_r`, discussion of the Table V comparison
- Corroboration: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`, comments preceding `case_dict`

## Sources

- [README.md](../../README.md) — `## Validation`.
- [Agent guide](../../.agents/README.md) — `## Scientific Standards`.
- [Cubic-in-spin Hamiltonian](../../nrpypn/PN_Hamiltonian_SSS.py) — `PN_Hamiltonian_SSS`, `f_H_SSS_3PN`.
- [Spin-spin Hamiltonian](../../nrpypn/PN_Hamiltonian_SS.py) — `PN_Hamiltonian_SS.f_H_SS_S1S2_3PN`.
- [Spin-orbit Hamiltonian](../../nrpypn/PN_Hamiltonian_SO.py) — `PN_Hamiltonian_SO.HS2011_Omega_SO_3p5PN_pt3`.
- [Tangential momentum](../../nrpypn/PN_p_t.py) — `PN_p_t.p_t__a_7`.
- [Orbital frequency](../../nrpypn/PN_MOmega.py) — `PN_MOmega.MOmega__a_7`.
- [Radial momentum](../../nrpypn/PN_p_r.py) — `PN_p_r.f_p_r`, `f_dr_dt`, `f_p_r_alternative__use_fullHam`.
- [Numerical evaluator](../../nrpypn/eval_p_t_and_p_r.py) — `test_results_against_trusted`, comments preceding `case_dict`.

## See Also

- Parent: [Validation](index.md)
- Depends on: [Scientific Evidence](scientific-evidence.md)
- See also: [Expression Regressions](expression-regressions.md)
- See also: [Cubic-In-Spin Hamiltonian](../equations/hamiltonians/cubic-in-spin.md)
- See also: [Spin-Orbit Hamiltonian](../equations/hamiltonians/spin-orbit.md)
- See also: [Radial Momentum](../equations/orbital-quantities/radial-momentum.md)
