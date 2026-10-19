# Catalog

> Find the owning page through topic, route, and query terms. · Status: confirmed
> Up: [NRPyPN Knowledge Base](../AGENTS.md)

## Summary

This inventory includes routers, sourced leaves, and support pages. Follow the
listed route and read the leaf before answering; a short catalog description
is not scientific evidence. Terms in the glossary already have defining pages;
new recurring concepts require a clear owner before becoming separate hubs.

## Pages

| Page | Type | One-line answer | Route | Tags | Aliases / Query terms | Status | Concept hub candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Knowledge Base Schema](SCHEMA.md) | support | Required page layouts, evidence, authority, and maintenance rules. | [Root](../AGENTS.md) | governance; page-layout | schema; status; source tiers; claim evidence | confirmed | no |
| [Architecture](architecture/index.md) | router | Choose the package overview or installation and packaging guidance. | [Root](../AGENTS.md) | navigation; architecture | package; install; dependencies | router | no |
| [Installation and Packaging](architecture/installation-and-packaging.md) | leaf | Distinguish installation instructions from declared package requirements. | [Root](../AGENTS.md) → [Architecture](architecture/index.md) | architecture; packaging | pip; setup.py; requirements; Python compatibility; py.typed | confirmed | no |
| [Calculation Overview](architecture/overview.md) | leaf | Trace symbolic expressions through numerical momentum evaluation. | [Root](../AGENTS.md) → [Architecture](architecture/index.md) | architecture; equations | module map; call sequence; package purpose; low eccentricity | confirmed | no |
| [Catalog](catalog.md) | support | Find every wiki page by topic, route, and query term. | [Root](../AGENTS.md) | navigation; catalog | page inventory; search; aliases | confirmed | no |
| [Language For Computational Physicists](computational-physics-language.md) | support | Write direct explanations of equations, computations, files, and evidence. | [Root](../AGENTS.md) | governance; prose | wording; language; terminology; comments; docstrings | confirmed | no |
| [Contradictions](contradictions.md) | support | Reconcile competing or stale claims with explicit deciding evidence. | [Root](../AGENTS.md) | maintenance; evidence | CONTR; contested; stale; resolved; authority | confirmed | no |
| [Cubic-in-Spin Hamiltonian](equations/hamiltonians/cubic-in-spin.md) | leaf | Distinguish the cubic-spin expression and corrected coefficient from its historical name. | [Root](../AGENTS.md) → [Equations](equations/index.md) → [Hamiltonians](equations/hamiltonians/index.md) | equations; Hamiltonians | H_SSS_3PN; 3.5PN; Levi; Steinhoff; Hergt; Schaefer; mass prefactor; -9/4; spin cubed | confirmed | no |
| [Hamiltonians](equations/hamiltonians/index.md) | router | Choose nonspinning, linear, quadratic, or cubic spin terms. | [Root](../AGENTS.md) → [Equations](equations/index.md) | navigation; Hamiltonians | NS; SO; SS; SSS; independent momenta | router | no |
| [Nonspinning Hamiltonian](equations/hamiltonians/nonspinning.md) | leaf | Locate nonspinning Hamiltonian terms and their reduced-variable convention. | [Root](../AGENTS.md) → [Equations](equations/index.md) → [Hamiltonians](equations/hamiltonians/index.md) | equations; Hamiltonians | H_Newt; H_NS_1PN; H_NS_2PN; H_NS_3PN; reduced mass | confirmed | no |
| [Spin-Orbit Hamiltonian](equations/hamiltonians/spin-orbit.md) | leaf | Locate spin-orbit terms that general momenta can expose. | [Root](../AGENTS.md) → [Equations](equations/index.md) → [Hamiltonians](equations/hamiltonians/index.md) | equations; Hamiltonians | H_SO; Omega; HS2011; p1 cross p2; centre of mass | confirmed | no |
| [Spin-Spin Hamiltonian](equations/hamiltonians/spin-spin.md) | leaf | Locate mixed-spin and self-spin Hamiltonian contributions. | [Root](../AGENTS.md) → [Equations](equations/index.md) → [Hamiltonians](equations/hamiltonians/index.md) | equations; Hamiltonians | H_SS_2PN; H_SS_S1S2_3PN; H_SS_S1sq_S2sq_3PN; S1 dot p2 | confirmed | no |
| [Equations](equations/index.md) | router | Choose notation, Hamiltonian terms, or orbital quantities. | [Root](../AGENTS.md) | navigation; equations | PN; symbols; spin; Hamiltonian; momentum; flux | router | no |
| [Energy Flux and Mass Change](equations/orbital-quantities/fluxes.md) | leaf | Follow energy and mass fluxes into the radial-drift calculation. | [Root](../AGENTS.md) → [Equations](equations/index.md) → [Orbital Quantities](equations/orbital-quantities/index.md) | equations; orbital-quantities | PN_dE_GW_dt_and_dM_dt; x; balance; absorption | confirmed | no |
| [Orbital Quantities](equations/orbital-quantities/index.md) | router | Choose tangential momentum and frequency, radial momentum, or fluxes. | [Root](../AGENTS.md) → [Equations](equations/index.md) | navigation; orbital-quantities | p_t; MOmega; p_r; dE_GW_dt; dM_dt | router | no |
| [Radial Momentum](equations/orbital-quantities/radial-momentum.md) | leaf | Follow inward radial drift and the closed and full-Hamiltonian momentum routes. | [Root](../AGENTS.md) → [Equations](equations/index.md) → [Orbital Quantities](equations/orbital-quantities/index.md) | equations; orbital-quantities | PN_p_r; dr_dt; alternative; matching PN order; in-plane spin | confirmed | no |
| [Tangential Momentum and Orbital Frequency](equations/orbital-quantities/tangential-momentum-and-frequency.md) | leaf | Locate the quasicircular tangential-momentum and frequency series. | [Root](../AGENTS.md) → [Equations](equations/index.md) → [Orbital Quantities](equations/orbital-quantities/index.md) | equations; orbital-quantities | PN_p_t; PN_MOmega; a_2; a_7; truncation | confirmed | no |
| [Symbols and Conventions](equations/symbols-and-conventions.md) | leaf | Separate mass ratio from separation and follow spin and momentum signs. | [Root](../AGENTS.md) → [Equations](equations/index.md) | equations; conventions | q; r; chi1U; S1U; n21U; p2U; inward drift; exact rational | confirmed | no |
| [Glossary](glossary.md) | support | Find canonical notation and its defining pages. | [Root](../AGENTS.md) | navigation; terminology | q; qmassratio; chi; Pr; trusted results | confirmed | no |
| [Lint Checks](lint/CHECKS.md) | support | Check Markdown structure, source use, and the inherited quality requirements. | [Root](../AGENTS.md) | governance; validation | KB lint; link checks; whitespace; Pylint | confirmed | no |
| [Source Map](source-map.md) | support | Follow exact source dependencies and bounded ingest coverage. | [Root](../AGENTS.md) | maintenance; evidence | reverse dependencies; registered; partial; exact seed | confirmed | no |
| [Command Line and TwoPunctures](usage/command-line-and-twopunctures.md) | leaf | Use the positional CLI and interpret its TwoPunctures sign mapping. | [Root](../AGENTS.md) → [Usage](usage/index.md) | usage; CLI | quasicirc; --help; par_b; par_P_plus; par_S_plus; no arguments | confirmed | no |
| [Usage](usage/index.md) | router | Choose the Python evaluator or CLI and TwoPunctures instructions. | [Root](../AGENTS.md) | navigation; usage | API; CLI; initial data | router | no |
| [Numerical Evaluation](usage/numerical-evaluation.md) | leaf | Evaluate tangential and radial momenta with explicit normalized inputs. | [Root](../AGENTS.md) → [Usage](usage/index.md) | usage; evaluation | eval__P_t__and__P_r; num_eval; qmassratio; floats | confirmed | no |
| [Expression Regressions](validation/expression-regressions.md) | leaf | Compare existing reference values without confusing generation with comparison. | [Root](../AGENTS.md) → [Validation](validation/index.md) | validation; regression | trusted_dict; compare_or_generate_trusted_results; DOCTEST_MODE; tolerance; NaN; finite momenta; fixed samples; symbol rename | confirmed | no |
| [Validation](validation/index.md) | router | Choose regression checks, scientific evidence, or static analysis and CI. | [Root](../AGENTS.md) | navigation; validation | testing; confidence; corrections | router | no |
| [Scientific Corrections](validation/scientific-corrections.md) | leaf | Follow documented corrections and the downstream quantities they affect. | [Root](../AGENTS.md) → [Validation](validation/index.md) | validation; science | SS; SO; SSS; in-plane spin; corrected coefficients | confirmed | no |
| [Scientific Evidence](validation/scientific-evidence.md) | leaf | Select evidence that can detect scientific errors missed by transcriptions. | [Root](../AGENTS.md) → [Validation](validation/index.md) | validation; science | independent derivation; symmetry; physical limits; publication agreement | confirmed | no |
| [Static Analysis And CI](validation/static-analysis-and-ci.md) | leaf | Apply scoped checks and distinguish required scores from configured CI behavior. | [Root](../AGENTS.md) → [Validation](validation/index.md) | validation; static-analysis | Black; mypy; Pylint; 10.00; 9.5; Python 3.6.7; standalone script | contested | no |
| [Workflows](workflows.md) | support | Register sources, ingest claims, answer queries, and reconcile changes. | [Root](../AGENTS.md) | governance; maintenance | ingest; query filing; source drift; safe reproduction | confirmed | no |

## Sources

- [Inherited KB conventions](../raw/source-docs/nrpy-kb-schema.md) — `## Support Pages`, `## Router Rules`, `## Canonical Terms`

## See Also

- Parent: [NRPyPN Knowledge Base](../AGENTS.md)
- Depends on: [Glossary](glossary.md)
- See also: [Workflows](workflows.md)
- See also: [Source Map](source-map.md)
