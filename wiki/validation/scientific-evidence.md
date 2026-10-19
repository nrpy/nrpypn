# Scientific Evidence

> Distinguish transcription agreement, stored-result regression, and evidence for an equation. · Status: confirmed
> Up: [Validation](index.md)

## Summary

A match to a printed expression establishes transcription agreement. A match to
stored values establishes agreement with that numerical reference. Neither
alone establishes scientific correctness. NRPyPN's README and affected module
docstrings document corrections that earlier comparisons missed; scientific
changes need evidence capable of detecting the particular error.

## Detail

### What A Comparison Establishes

The README distinguishes comparison with the cited expressions from proof that
those expressions are correct. Two transcriptions of the same printed formula
can share its error. Regenerated trusted values likewise preserve the selected
implementation rather than independently derive it. References in docstrings
are bibliographic attributions; this page describes the implementation rather
than independently verifying its derivation.

Claim evidence:
- Claim: Agreement with a publication, another transcription, or regenerated stored values does not independently establish the physical correctness of the shared expression.
- Role: public/scientific contract
- Deciding authority: [README.md](../../README.md), `## Validation`
- Corroboration: [agent guide](../../.agents/README.md), `## Scientific Standards`; [PN_Hamiltonian_SSS.py](../../nrpypn/PN_Hamiltonian_SSS.py), `PN_Hamiltonian_SSS` docstring

| Evidence | Supports | Limit |
| --- | --- | --- |
| Agreement with a named printed equation | Transcription agreement under the stated conventions | A printed error can survive unchanged. |
| Agreement with stored expression dictionaries | Stability of the expressions' numerical evaluations | Shared derivation errors and untested substitutions remain possible. |
| Agreement with evaluator cases | Numerical momentum behavior at the specified masses, spins, and separations | Restricted cases can hide terms that vanish for those inputs. |
| Independent derivation, symbolic identity, Hamilton's equations, exchange symmetry, or a physical limit | The specific mathematical relation or physical requirement examined | Its assumptions and PN order determine what it can decide. |
| Workflow configuration | Which checks CI is configured to execute | Configuration does not establish a successful execution. |

Claim evidence:
- Claim: Each evidence type supports only its stated comparison, mathematical relation, or configured checks under the specified assumptions; shared transcription and numerical references do not supply an independent derivation.
- Role: public/scientific contract
- Deciding authority: [agent guide](../../.agents/README.md), `## Scientific Standards` and `## Validation`
- Corroboration: [README.md](../../README.md), `## Validation`; [main.yml](../../.github/workflows/main.yml), `jobs.build`, configured checks

The comparison routes are detailed in [Expression
Regressions](expression-regressions.md). The correction examples belong in
[Scientific Corrections](scientific-corrections.md), rather than being treated
as unresolved disagreements between the corrected implementation and its
stored values.

### Evidence Required For Equation Changes

For a formula change, identify the publication and equation, normalization,
units, spin convention, and PN truncation. Keep coefficients and exponents
exact with `div(...)` or `sp.Rational`. Choose mathematical evidence that can
detect the suspected error: an independent derivation, symbolic identity,
Hamilton's equations, exchange symmetry, or a known physical limit. Numerical
sampling supports that reasoning; it does not replace it.

Claim evidence:
- Claim: Scientific changes require explicit conventions, exact symbolic coefficients, and mathematical evidence capable of detecting the suspected error; numerical sampling alone is insufficient.
- Role: normative rule
- Deciding authority: [agent guide](../../.agents/README.md), `## Scientific Standards`
- Corroboration: [README.md](../../README.md), `## Validation`, examples of errors missed by transcription comparisons

Check a general Hamiltonian with independent `p1U` and `p2U` before imposing
centre-of-mass or circular-orbit substitutions. Include unequal masses and
in-plane spins where the changed term needs them. A term proportional to
`p1U` cross `p2U` disappears when `p2U = -p1U`; a circular-orbit restriction
can likewise remove terms whose radial-momentum derivatives still matter.

Claim evidence:
- Claim: General-Hamiltonian validation must precede restrictive momentum substitutions, because centre-of-mass and circular-orbit restrictions can hide terms relevant to the general expression or its derivatives.
- Role: normative rule
- Deciding authority: [agent guide](../../.agents/README.md), `## Scientific Standards`
- Corroboration: [PN_Hamiltonian_SO.py](../../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO.HS2011_Omega_SO_3p5PN_pt3`; [PN_Hamiltonian_SS.py](../../nrpypn/PN_Hamiltonian_SS.py), `PN_Hamiltonian_SS.f_H_SS_S1S2_3PN`

### Follow The Changed Quantity

Trace a Hamiltonian correction through the quantities that use it: tangential
momentum, orbital frequency, fluxes, radial velocity, radial momentum,
numerical evaluation, and their stored references. Check both radial-momentum
approaches when affected. Compare them at matching PN order: the
full-Hamiltonian route retains higher-order terms, so exact numerical equality
with the truncated closed form is not a general requirement.

Claim evidence:
- Claim: Scientific changes require dependency checks and, when affected, both radial-momentum routes at matching PN order; exact untruncated equality between those routes is not generally expected.
- Role: normative rule
- Deciding authority: [agent guide](../../.agents/README.md), `## Scientific Standards`
- Corroboration: [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r` class docstring, `f_dr_dt`, `f_p_r_alternative__use_fullHam`, and `f_p_r`

Stored results may be updated only with independent scientific justification
and an explanation of the affected expressions and physical cases. Do not
delete reference values, relax tolerances, remove assertions, or add
suppressions merely to obtain a passing check.

Claim evidence:
- Claim: A reference update needs independent scientific justification; weakening a check or regenerating values solely to pass is prohibited.
- Role: normative rule
- Deciding authority: [AGENTS.md](../../AGENTS.md), `## Python Quality Requirements`
- Corroboration: [agent guide](../../.agents/README.md), `## Scientific Standards`

## Sources

- [README.md](../../README.md) — `## Validation`.
- [Agent guide](../../.agents/README.md) — `## Scientific Standards`, `## Validation`.
- [AGENTS.md](../../AGENTS.md) — `## Python Quality Requirements`.
- [Cubic-in-spin Hamiltonian](../../nrpypn/PN_Hamiltonian_SSS.py) — `PN_Hamiltonian_SSS` docstring.
- [Spin-orbit Hamiltonian](../../nrpypn/PN_Hamiltonian_SO.py) — `PN_Hamiltonian_SO.HS2011_Omega_SO_3p5PN_pt3`.
- [Spin-spin Hamiltonian](../../nrpypn/PN_Hamiltonian_SS.py) — `PN_Hamiltonian_SS.f_H_SS_S1S2_3PN`.
- [Radial momentum](../../nrpypn/PN_p_r.py) — `PN_p_r`, `f_dr_dt`, `f_p_r_alternative__use_fullHam`, `f_p_r`.
- [CI workflow](../../.github/workflows/main.yml) — `jobs.build`, configured checks.

## See Also

- Parent: [Validation](index.md)
- Contrasts with: [Expression Regressions](expression-regressions.md)
- See also: [Scientific Corrections](scientific-corrections.md)
- Depends on: [Symbols And Conventions](../equations/symbols-and-conventions.md)
- See also: [Radial Momentum](../equations/orbital-quantities/radial-momentum.md)
