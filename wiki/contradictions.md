# Contradictions

> Track competing claims, deciding evidence, and the pages requiring reconciliation. · Status: confirmed
> Up: [NRPyPN Knowledge Base](../AGENTS.md)

## Summary

Use this register when sources disagree or a living-source change leaves a
compiled claim unreliable. A reported historical equation correction is not
automatically an active source conflict. The scientific pages distinguish the
implemented expression, the module's account of a correction, and evidence
independent of either transcription.

## Register

| ID | Claim | Claim status | Source A | Source B | Authority decision | Affected pages | Page-status rationale | Owner/trigger | Resolution test | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CONTR-0001 | Configured score parsing does not enforce new-file Pylint 10.00 or existing-file no-regression requirements. | contested | [Root rule](../AGENTS.md), `Python Quality Requirements` | [Workflow](../.github/workflows/main.yml), `jobs.build` score comparison; [standalone script](../.github/standalone_bash_script.sh), `pylint_score` comparison | Root governance decides the required scores; parsed-score floors describe enforcement only. Apply the rule separately. | [Static Analysis And CI](validation/static-analysis-and-ci.md) | Page is contested because the enforcement gap is central to choosing sufficient checks. | Python quality maintainer; revisit when score classification or parsing changes. | Inspect both loops for new/existing classification, compare existing files under identical settings, reject missing ratings, and require new 10.00 plus no legacy regression. | The protected workflow needs exact-file authorization before any enforcement change. |
| CONTR-0002 | num_eval is annotated float but can return a SymPy object or retain free symbols. | contested | [NRPyPN_shortcuts.py](../nrpypn/NRPyPN_shortcuts.py), `num_eval` return annotation | [NRPyPN_shortcuts.py](../nrpypn/NRPyPN_shortcuts.py), `num_eval` substituted expression returned through `cast` | The actual return expression decides behavior; no lower-level Python-float guarantee is asserted. The outer evaluator converts explicitly. | [Numerical Evaluation](usage/numerical-evaluation.md) | Page stays confirmed: the main momentum API converts both outputs; one bounded helper type discrepancy is qualified in Detail. | num_eval maintainer; revisit when return conversion or annotation changes. | Check num_eval on a SymPy integer and an unregistered free symbol; require an accurate declared return type or explicit conversion with defined rejection of unresolved symbols. Retain outer-evaluator float checks. | Optional-zero substitution is separately documented as an implementation limit. |
| CONTR-0003 | The nonspinning class documents nU from body 1 to body 2, opposite to the shared caller's nU = n12U convention. | contested | [PN_Hamiltonian_NS.py](../nrpypn/PN_Hamiltonian_NS.py), `PN_Hamiltonian_NS` parameter docstring | [Shortcuts](../nrpypn/NRPyPN_shortcuts.py), `n12U = nU`; [SO](../nrpypn/PN_Hamiltonian_SO.py), `n12U` definition; [radial calculation](../nrpypn/PN_p_r.py), `f_Htot_xyplane_binary` | The shared substitutions decide the assembled calculation's direction. Distinguish the class wording; no nonspinning sign error is established by this discrepancy alone. | [Nonspinning Hamiltonian](equations/hamiltonians/nonspinning.md) | Page stays confirmed: one bounded direction-description discrepancy does not change the stated expressions or reduced-momentum convention. | Nonspinning Hamiltonian maintainer; revisit on direction documentation or directional-expression changes. | Trace nU through the shared caller, make the class convention explicit, and check each stored Hamiltonian under nU replaced by -nU before extending the convention to signed terms. | Do not change an equation merely to align its direction wording. |
| CONTR-0004 | The self-spin method's spin-orbit label conflicts with its quadratic-spin expression. | contested | [SS docstring](../nrpypn/PN_Hamiltonian_SS.py), `f_H_SS_S1sq_S2sq_3PN` | [SS expression](../nrpypn/PN_Hamiltonian_SS.py), nested `f_H_SS_particle` and `H_SS_S1sq_S2sq_3PN` assignment | Spin degree in the defining expression decides current classification; the prose label does not change the equation. | [Spin-Spin Hamiltonian](equations/hamiltonians/spin-spin.md) | Page stays confirmed: the bounded label defect is qualified and the named expression is described accurately. | Self-spin method maintainer; revisit when its docstring or expression changes. | Inspect the method label and substitute a common spin scale lambda; confirm the expression scales as lambda squared and the description names self-spin terms. | This row does not independently verify the cited publication. |
| CONTR-0005 | The SO class broadly attributes terms through 3.5PN to DJS, while the 3.5PN method attributes its term to Hartung and Steinhoff. | contested | [SO class](../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO` docstring | [SO method](../nrpypn/PN_Hamiltonian_SO.py), `f_H_SO_3p5PN` docstring | Use per-method references for the repository's attribution; external derivation remains outside this evidence. | [Spin-Orbit Hamiltonian](equations/hamiltonians/spin-orbit.md) | Page stays confirmed: one bounded attribution discrepancy is qualified; the stored terms and general-momenta behavior remain supported. | SO documentation maintainer; revisit when reference text changes. | Compare the class attribution with each defining method and make the class description match those explicit references; independently consult a stable publication before claiming scientific attribution is verified. | Bibliographic reconciliation is distinct from checking coefficients. |
| CONTR-0006 | The workflow's original-implementation reference label overstates the current provenance of the spinning-case values. | contested | [Workflow](../.github/workflows/main.yml), final validation step name | [Evaluator](../nrpypn/eval_p_t_and_p_r.py), comments preceding `case_dict` in `test_results_against_trusted` | The test source documents regenerated spinning references after corrections; interpret the workflow label as historical wording rather than an unchanged-reference guarantee. | [Static Analysis And CI](validation/static-analysis-and-ci.md) | The label is one bounded Detail discrepancy; the page is already contested for the central score-enforcement gap. | Validation-step documentation maintainer; revisit when the label or reference provenance changes. | Compare the workflow label with the evaluator's reference-data comments and documented corrections; require wording that accurately identifies the current regression values without implying an independent original reference. | Workflow text remains protected by exact-file authorization. |


## Reconciliation Rules

Assign immutable `CONTR-0001`-form identifiers. Active rows use `contested` or
`stale`; reconciled rows use `resolved`. Follow the page-status matrix in
[KB Page Format](SCHEMA.md#page-status-decision-matrix): the status of one
claim need not be the status of its whole page. Link every affected leaf and
place the exact active marker and backlink there. Put the claim's evidence
block under its matching `### CONTR-*` subsection here, outside the table.

Describe the deciding source for the type of claim: implementation for current
behavior, governing instructions for a normative rule, and a stable scientific
specification with targeted mathematical or numerical checks for an intended
scientific requirement. Neighboring wiki pages are not independent evidence.

Before resolving a row, inspect reverse dependencies in
[Source Map](source-map.md), all pages citing either source, catalog aliases,
the linked affected pages, typed neighbors, and targeted phrase matches.
Require a deterministic inspection or executable test that can decide the
disagreement. Preserve the bounded conditions and remaining uncertainty in the
owning leaf; keep transient check results in command output or review evidence.

## Claim Evidence

### CONTR-0001

The contributor score requirements exceed the configured parsed-score checks.

Claim evidence:
- Claim: The configured workflow and standalone script compare a parsed score with 9.5, without new-file classification or pre-change score comparison, so they do not enforce the adopted score distinction.
- Role: CI behavior
- Deciding authority: [Workflow](../.github/workflows/main.yml), `jobs.build` inline `pylint_score` comparison; [standalone script](../.github/standalone_bash_script.sh), `pylint_score` comparison. [AGENTS.md](../AGENTS.md), `## Python Quality Requirements`, decides the normative requirement.
- Corroboration: [.pylintrc](../.pylintrc) and [.pylintrc_python36](../.pylintrc_python36), `fail-under=10`, specify a blanket threshold rather than new/existing classification; they do not establish successful checks.

### CONTR-0002

The lower-level annotation must not be promoted into a runtime conversion guarantee.

Claim evidence:
- Claim: num_eval's substituted SymPy expression is returned through typing.cast despite the float annotation; arbitrary free symbols need not be removed, and the result need not be a Python float.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../nrpypn/NRPyPN_shortcuts.py), `num_eval` annotation and return expression.
- Corroboration: [eval_p_t_and_p_r.py](../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`, explicitly uses float on both outputs; this establishes the outer conversion only. The resolution inputs above target the lower-level discrepancy.

### CONTR-0003

Direction names must be interpreted with their defining caller.

Claim evidence:
- Claim: The nonspinning class describes nU from body 1 to body 2, while the assembled radial caller uses the shared nU = n12U and the spin Hamiltonian definition of n12U from body 2 to body 1; this documents differing direction conventions without establishing a nonspinning sign error.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_NS.py](../nrpypn/PN_Hamiltonian_NS.py), `PN_Hamiltonian_NS` parameter docstring; [NRPyPN_shortcuts.py](../nrpypn/NRPyPN_shortcuts.py), derived `n12U`; [PN_Hamiltonian_SO.py](../nrpypn/PN_Hamiltonian_SO.py), `n12U` parameter definition.
- Corroboration: [PN_p_r.py](../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary`, passes the shared nU and sets its instantaneous direction through `make_replacements`.

### CONTR-0004

Expression degree decides whether the self-spin method is linear or quadratic in spin.

Claim evidence:
- Claim: f_H_SS_S1sq_S2sq_3PN calls its term spin-orbit in the docstring, while its expression contains quadratic self-spin factors and is added separately from SO terms.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SS.py](../nrpypn/PN_Hamiltonian_SS.py), `f_H_SS_S1sq_S2sq_3PN` docstring and nested `f_H_SS_particle`.
- Corroboration: [PN_p_r.py](../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary`, assembles self-spin attributes separately from SO attributes.

### CONTR-0005

The broad class attribution does not replace the individual method references.

Claim evidence:
- Claim: The SO class attributes Hamiltonians through 3.5PN broadly to DJS, whereas f_H_SO_3p5PN names Hartung and Steinhoff; this is a repository attribution discrepancy, not independently verified bibliography.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SO.py](../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO` and `f_H_SO_3p5PN` docstrings.
- Corroboration: none available for independent bibliographic verification; comparing the defining docstrings establishes their differing attributions only.

### CONTR-0006

A configured check's label must not be mistaken for independent reference provenance.

Claim evidence:
- Claim: The workflow labels its final momentum test as comparison with original-implementation values, while the defining evaluator comments document regenerated spinning references after coefficient and sign corrections; the label does not establish unchanged or independent references.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`, comments preceding `case_dict`; [workflow](../.github/workflows/main.yml), final validation step name, supplies the competing wording.
- Corroboration: [README.md](../README.md), `## Validation`, reports the related inherited errors and corrections; it is a separate repository account, not an independent derivation.

## Sources

- [Inherited KB conventions](../raw/source-docs/nrpy-kb-schema.md) — `## Contradiction Contract`, `### Page Status Decision Matrix`, `### Claim And Evidence Contract`
- [README.md](../README.md) — `## Validation`, historical scientific corrections and the limits of transcription comparisons

- [AGENTS.md](../AGENTS.md) — `## Python Quality Requirements`
- [Workflow](../.github/workflows/main.yml) — `jobs.build`, `pylint_score` comparison
- [Standalone script](../.github/standalone_bash_script.sh) — `pylint_score` comparison
- [.pylintrc](../.pylintrc) — `fail-under`
- [.pylintrc_python36](../.pylintrc_python36) — `fail-under`
- [NRPyPN_shortcuts.py](../nrpypn/NRPyPN_shortcuts.py) — `num_eval`, shared `n12U`
- [eval_p_t_and_p_r.py](../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r`
- [PN_Hamiltonian_NS.py](../nrpypn/PN_Hamiltonian_NS.py) — `PN_Hamiltonian_NS` docstring
- [PN_Hamiltonian_SO.py](../nrpypn/PN_Hamiltonian_SO.py) — class and `f_H_SO_3p5PN` docstrings
- [PN_Hamiltonian_SS.py](../nrpypn/PN_Hamiltonian_SS.py) — `f_H_SS_S1sq_S2sq_3PN`
- [PN_p_r.py](../nrpypn/PN_p_r.py) — `f_Htot_xyplane_binary`

## See Also

- Parent: [NRPyPN Knowledge Base](../AGENTS.md)
- Depends on: [KB Page Format](SCHEMA.md)
- Depends on: [Source Map](source-map.md)
- See also: [Scientific Evidence](validation/scientific-evidence.md)
