# Glossary

> Canonical terms and the pages that define their use in NRPyPN. · Status: confirmed
> Up: [NRPyPN Knowledge Base](../AGENTS.md)

## Summary

Use these names consistently across equations, numerical evaluation, and
validation. Follow the defining page for assumptions and evidence; this table
is a navigation aid.

## Terms

| Term | Meaning |
| --- | --- |
| NRPyPN | SymPy post-Newtonian expressions and quasicircular momentum calculations; see [Architecture](architecture/overview.md). |
| NRPy | Dependency supplying indexed-expression and validation helpers; see [Installation And Packaging](architecture/installation-and-packaging.md). |
| PN | Post-Newtonian expansion, with truncation and spin counting stated for the quantity; see [Symbols And Conventions](equations/symbols-and-conventions.md). |
| NS | Nonspinning Hamiltonian terms; see [Nonspinning Hamiltonian](equations/hamiltonians/nonspinning.md). |
| SO | Spin-orbit terms, linear in spin; see [Spin-Orbit Hamiltonian](equations/hamiltonians/spin-orbit.md). |
| SS | Quadratic-in-spin terms, including self-spin terms; see [Spin-Spin Hamiltonian](equations/hamiltonians/spin-spin.md). |
| SSS | Cubic-in-spin terms; their physical PN order must be distinguished from the historical method name; see [Cubic-In-Spin Hamiltonian](equations/hamiltonians/cubic-in-spin.md). |
| `qmassratio` | Numerical evaluator mass ratio `m2/m1`; see [Numerical Evaluation](usage/numerical-evaluation.md). |
| `q`, `r` | Separation symbols in the symbolic calculations, not automatically a mass ratio; see [Symbols And Conventions](equations/symbols-and-conventions.md). |
| `chi1U`, `chi2U` | Dimensionless spin components, related to physical spins by `S_i = chi_i*m_i**2`; see [Symbols And Conventions](equations/symbols-and-conventions.md). |
| Centre-of-mass substitution | Opposite canonical momenta used in the orbital calculation, which can hide terms in a general Hamiltonian; see [Spin-Orbit Hamiltonian](equations/hamiltonians/spin-orbit.md). |
| `p_t` | Tangential canonical momentum; see [Tangential Momentum And Frequency](equations/orbital-quantities/tangential-momentum-and-frequency.md). |
| `MOmega` | Total-mass-scaled orbital angular frequency; see [Tangential Momentum And Frequency](equations/orbital-quantities/tangential-momentum-and-frequency.md). |
| `p_r`, `dr_dt` | Radial momentum and inward-positive radial drift convention; see [Radial Momentum](equations/orbital-quantities/radial-momentum.md). |
| Flux balance | Energy-loss and mass-change terms used in radial drift; see [Fluxes](equations/orbital-quantities/fluxes.md). |
| TwoPunctures | Initial-data input mapping printed by the command-line interface; see [Command Line And TwoPunctures](usage/command-line-and-twopunctures.md). |
| Trusted results | Stored high-precision expression values used for regression comparison; see [Expression Regressions](validation/expression-regressions.md). |
| Scientific evidence | Derivations, identities, symmetries, and physical limits that test the intended equations; see [Scientific Evidence](validation/scientific-evidence.md). |
| Frozen source | Verbatim imported evidence, distinct from compiled prose; see [KB Page Format](SCHEMA.md#ownership). |
| Living source | Repository code or documentation that may change and require dependent-page reconciliation; see [Workflows](workflows.md#living-source-change). |
| Claim status | Whether a particular assertion is contested, stale, or resolved; distinct from the overall page status; see [Contradictions](contradictions.md). |

## Sources

- [NRPyPN_shortcuts.py](../nrpypn/NRPyPN_shortcuts.py) — `num_eval`, `div`, shared symbols
- [eval_p_t_and_p_r.py](../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r`
- [Inherited KB conventions](../raw/source-docs/nrpy-kb-schema.md) — `## Canonical Terms`, `## Ownership`

## See Also

- Parent: [NRPyPN Knowledge Base](../AGENTS.md)
- Depends on: [Catalog](catalog.md)
- See also: [Language For Computational Physicists](computational-physics-language.md)
