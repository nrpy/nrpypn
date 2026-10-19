# Expression Regressions

> Locate stored expression comparisons and physical-case momentum regressions. · Status: confirmed
> Up: [Validation](index.md)

## Summary

NRPyPN checks symbolic calculations by numerically evaluating their expressions
and comparing the resulting dictionaries with stored values. The evaluator also
contains numerical binary-black-hole cases for tangential and radial momentum.
These comparisons detect changes relative to stored results; they do not
independently establish the derivation of an expression. With the preserved
dependency helper, missing trusted data can be generated instead of compared,
including with `DOCTEST_MODE=1`.

## Detail

### Executable Module Checks

The equation modules' `__main__` blocks run `doctest.testmod()`, exit on doctest
failure, construct symbolic calculation objects, and pass their `__dict__` to
`ve.process_dictionary_of_expressions` with
`fixed_mpfs_for_free_symbols=True`. They then call
`ve.compare_or_generate_trusted_results` with the module path, working
directory, a trusted-file basename, and the numerical results. This is a
regression of symbolic expressions through numerical evaluation, rather than
an exact symbolic identity check.

Claim evidence:
- Claim: Equation-module execution combines doctests with numerical evaluation of symbolic attributes and a compare-or-generate call; it does not perform an independent symbolic derivation.
- Role: descriptive behavior
- Deciding authority: [PN_p_t.py](../../nrpypn/PN_p_t.py), `__main__`; [PN_Hamiltonian_NS.py](../../nrpypn/PN_Hamiltonian_NS.py), `__main__`
- Corroboration: [PN_MOmega.py](../../nrpypn/PN_MOmega.py), `__main__`; [agent guide](../../.agents/README.md), `## Validation`

In the [preserved sample-selection definition](../../raw/source-docs/nrpy-expression-sampling.md),
`convert_free_symbols_set_to_mpf_dict` with
`fixed_mpfs_for_free_symbols=True` seeds each ordinary symbol's numerical
sample from its string representation. The named pi and inverse-square-root
constants take separate fixed values. Renaming a free symbol can therefore
change the sampled point and stored numerical reference even when the
expressions differ only by that rename. Such a difference is not by itself
evidence of a changed equation and does not justify regenerating references.

Claim evidence:
- Claim: In the preserved sample-selection implementation, fixed samples for ordinary free symbols depend on symbolic names, so a rename can change numerical references without changing the mathematical expression up to that rename.
- Role: descriptive behavior
- Deciding authority: [Preserved sample selection](../../raw/source-docs/nrpy-expression-sampling.md), `## convert_free_symbols_set_to_mpf_dict`
- Corroboration: none available for behavior across dependency releases; the excerpt defines only the preserved implementation and the imported helper must be inspected when the unpinned dependency changes.

The stored `trusted_dict` names identify which attributes are compared:

| Calculation | Stored source and stable locator |
| --- | --- |
| Nonspinning Hamiltonian | [PN_Hamiltonian_NS.py](../../nrpypn/tests/PN_Hamiltonian_NS.py), `trusted_dict`: `H_Newt`, `H_NS_1PN`, `H_NS_2PN`, `H_NS_3PN` |
| Spin-orbit Hamiltonian | [PN_Hamiltonian_SO.py](../../nrpypn/tests/PN_Hamiltonian_SO.py), `trusted_dict`: `H_SO_1p5PN`, `H_SO_2p5PN`, `H_SO_3p5PN` |
| Quadratic-in-spin Hamiltonian | [PN_Hamiltonian_SS.py](../../nrpypn/tests/PN_Hamiltonian_SS.py), `trusted_dict`: `H_SS_2PN`, `H_SS_S1S2_3PN`, `H_SS_S1sq_S2sq_3PN` |
| Cubic-in-spin Hamiltonian | [PN_Hamiltonian_SSS.py](../../nrpypn/tests/PN_Hamiltonian_SSS.py), `trusted_dict`: `H_SSS_3PN` |
| Orbital frequency | [PN_MOmega.py](../../nrpypn/tests/PN_MOmega.py), `trusted_dict`: `MOmega` and `a_2` through `a_7` |
| Tangential momentum | [PN_p_t.py](../../nrpypn/tests/PN_p_t.py), `trusted_dict`: `p_t` and `a_2` through `a_7` |
| Energy and mass fluxes | [PN_dE_GW_dt_and_dM_dt.py](../../nrpypn/tests/PN_dE_GW_dt_and_dM_dt.py), `trusted_dict`: `dE_GW_dt`, `dM_dt`, `dE_GW_dt_plus_dM_dt` |
| Closed-form radial momentum | [PN_p_r_AlternativeFalse.py](../../nrpypn/tests/PN_p_r_AlternativeFalse.py), `trusted_dict`: `Htot_xyplane_binary`, `dr_dt`, `p_r` |
| Full-Hamiltonian radial momentum | [PN_p_r_AlternativeTrue.py](../../nrpypn/tests/PN_p_r_AlternativeTrue.py), `trusted_dict`: `Htot_xyplane_binary`, `dr_dt`, `p_r` |

These Python files store `mpf` and sometimes `mpc` values. They are reference
data, not a pytest suite. Complex values in a symbolic-expression dictionary
must not be presented as physical binary-black-hole outputs; the separately
specified physical cases are in the numerical evaluator.

Claim evidence:
- Claim: The files under `nrpypn/tests/` store numerical expression reference data, including complex values, rather than executable physical-case tests.
- Role: descriptive behavior
- Deciding authority: [PN_p_r_AlternativeFalse.py](../../nrpypn/tests/PN_p_r_AlternativeFalse.py), `trusted_dict`; [PN_dE_GW_dt_and_dM_dt.py](../../nrpypn/tests/PN_dE_GW_dt_and_dM_dt.py), `trusted_dict`
- Corroboration: [agent guide](../../.agents/README.md), `## Source Map` and `## Validation`; [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`

### Missing Data And Radial-Momentum Branches

NRPyPN delegates comparison and file creation to
`nrpy.validate_expressions.validate_expressions`. In the preserved helper
implementation, `compare_or_generate_trusted_results` calls
`compare_against_trusted` when the trusted path exists and `output_trusted`
when it is missing. The latter writes a Python `trusted_dict` file under the
module's `tests/` directory. Neither helper checks `DOCTEST_MODE`, and the PN
module callers add no such guard. Thus `DOCTEST_MODE=1` does not prevent
missing-file generation with this implementation.

Claim evidence:
- Claim: In the preserved delegated helper implementation, a missing trusted path selects file generation without a `DOCTEST_MODE` check; PN module callers do not prevent that generation.
- Role: descriptive behavior
- Deciding authority: [Preserved NRPy helper definitions](../../raw/source-docs/nrpy-expression-validation.md), `## compare_or_generate_trusted_results` and `## output_trusted`
- Corroboration: [PN_p_t.py](../../nrpypn/PN_p_t.py), `__main__`, unguarded delegated call; [agent guide](../../.agents/README.md), `## Validation`, reference-data write warning

Require the expected dictionaries before claiming a comparison, execute in an
isolated copy with owned output and cache, and inspect the diff afterward.
Generating reference values is not evidence of agreement with pre-existing
values. Scope the helper description to the preserved implementation:
`requirements.txt` leaves `nrpy` unpinned, so inspect the imported helper when
that dependency changes.

Claim evidence:
- Claim: Comparison requires pre-existing reference data and isolated execution; the unpinned `nrpy` dependency requires inspection of the imported helper on dependency drift rather than assuming all releases match the preserved definitions.
- Role: normative rule
- Deciding authority: [agent guide](../../.agents/README.md), `## Validation`, imported-dependency check and reference-data write warning; [AGENTS.md](../../AGENTS.md), `## Python Quality Requirements`
- Corroboration: [requirements.txt](../../requirements.txt), unpinned `nrpy` entry; [preserved helper definitions](../../raw/source-docs/nrpy-expression-validation.md), document scope and `## compare_or_generate_trusted_results`

`PN_p_r.py` executes both values of
`use_alternative_approach_for_p_r` and selects the corresponding
`PN_p_r_AlternativeFalse` or `PN_p_r_AlternativeTrue` basename. It compares
each branch with its own stored dictionary; the loop does not assert equality
between the branches.

Claim evidence:
- Claim: Running `PN_p_r.py` checks the closed-form and full-Hamiltonian routes against separate trusted dictionaries, without asserting that the routes equal each other.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../nrpypn/PN_p_r.py), `__main__`, `for Alternative in [False, True]`
- Corroboration: [PN_p_r_AlternativeFalse.py](../../nrpypn/tests/PN_p_r_AlternativeFalse.py) and [PN_p_r_AlternativeTrue.py](../../nrpypn/tests/PN_p_r_AlternativeTrue.py), `trusted_dict`

### Physical-Case Momentum Regression

`test_results_against_trusted()` stores mass ratios, separations, spins, and
expected momenta in its local `case_dict`. It evaluates
`eval__P_t__and__P_r` and raises `ValueError` when either relative error
exceeds `1e-13`. The inputs include nonspinning, aligned-spin, and in-plane-spin
cases. The evaluator uses the default closed-form radial-momentum route, so
this test does not replace the full-Hamiltonian module comparison.

Claim evidence:
- Claim: The evaluator's physical-case regression checks both momenta with relative tolerance `1e-13` and uses the default closed-form radial-momentum calculation.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted` and `eval__P_t__and__P_r`
- Corroboration: [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.__init__`, default `use_alternative_approach_for_p_r=False`

The comparison has no finite-output check. Both rejection predicates are
`rel_error > tol`; NaN makes that comparison false. Thus the function can
return normally when evaluated momenta are NaN. Its tolerance comparison
alone does not establish finite numerical agreement.

Claim evidence:
- Claim: The physical-case comparison accepts NaN momenta because its ordered relative-error predicates are false for NaN; this does not establish NaN outputs for valid current inputs or a defect in the separate expression-dictionary comparison.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`, `rel_error_p_t`, `rel_error_p_r`, and their rejection predicates
- Corroboration: none available from an independent comparison; the defining predicates establish this detection limit.

For complete physical-case regression evidence, require finite evaluated
momenta before interpreting the tolerance check as agreement. In the isolated
NRPyPN copy, the following wraps every evaluator call made by the existing
case loop and restores the original evaluator afterward:

```python
from math import isfinite
from unittest.mock import patch
import nrpypn.eval_p_t_and_p_r as evaluation

evaluate = evaluation.eval__P_t__and__P_r


def evaluate_finite(*args, **kwargs):
    momenta = evaluate(*args, **kwargs)
    if not all(isfinite(value) for value in momenta):
        raise ValueError("Physical-case momenta must be finite")
    return momenta


with patch.object(evaluation, "eval__P_t__and__P_r", evaluate_finite):
    evaluation.test_results_against_trusted()
```

Claim evidence:
- Claim: Physical-case regression evidence requires finite momenta for every evaluated case before interpreting the existing relative-error comparison as agreement; the wrapper leaves the stored data and comparison implementation unchanged.
- Role: normative rule
- Deciding authority: [agent guide](../../.agents/README.md), `## Validation`, finite-output requirement
- Corroboration: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`, evaluator calls and relative-error predicates

The comments beside `case_dict` state that spinning-case values were
regenerated after the cubic-in-spin and in-plane-spin-sign corrections.
Agreement with those values checks the corrected implementation's stored
expectations; it is not an independent reproduction of the original tutorial
or of a published derivation.

Claim evidence:
- Claim: The spinning physical-case values incorporate documented corrections, so their historical origin does not make them an independent scientific reference.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`, comments immediately preceding `case_dict`
- Corroboration: [README.md](../../README.md), `## Validation`

## Sources

- [Agent guide](../../.agents/README.md) — `## Validation`, `## Scientific Standards`, `## Source Map`.
- [AGENTS.md](../../AGENTS.md) — `## Python Quality Requirements`.
- [Preserved NRPy helper definitions](../../raw/source-docs/nrpy-expression-validation.md) — `## compare_or_generate_trusted_results`, `## output_trusted`; selected dependency behavior only.
- [Preserved sample selection](../../raw/source-docs/nrpy-expression-sampling.md) — `## convert_free_symbols_set_to_mpf_dict`; selected dependency behavior only.
- [requirements.txt](../../requirements.txt) — unpinned `nrpy` dependency.
- [README.md](../../README.md) — `## Validation`.
- [PN_Hamiltonian_NS.py](../../nrpypn/PN_Hamiltonian_NS.py) — `__main__`.
- [PN_p_t.py](../../nrpypn/PN_p_t.py) — `__main__`.
- [PN_MOmega.py](../../nrpypn/PN_MOmega.py) — `__main__`.
- [PN_p_r.py](../../nrpypn/PN_p_r.py) — `PN_p_r.__init__`, `__main__`.
- [Numerical evaluator](../../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r`, `test_results_against_trusted`.
- [Nonspinning results](../../nrpypn/tests/PN_Hamiltonian_NS.py) — `trusted_dict`.
- [Spin-orbit results](../../nrpypn/tests/PN_Hamiltonian_SO.py) — `trusted_dict`.
- [Spin-spin results](../../nrpypn/tests/PN_Hamiltonian_SS.py) — `trusted_dict`.
- [Cubic-in-spin results](../../nrpypn/tests/PN_Hamiltonian_SSS.py) — `trusted_dict`.
- [Orbital-frequency results](../../nrpypn/tests/PN_MOmega.py) — `trusted_dict`.
- [Tangential-momentum results](../../nrpypn/tests/PN_p_t.py) — `trusted_dict`.
- [Flux results](../../nrpypn/tests/PN_dE_GW_dt_and_dM_dt.py) — `trusted_dict`.
- [Closed-form radial-momentum results](../../nrpypn/tests/PN_p_r_AlternativeFalse.py) — `trusted_dict`.
- [Full-Hamiltonian radial-momentum results](../../nrpypn/tests/PN_p_r_AlternativeTrue.py) — `trusted_dict`.

## See Also

- Parent: [Validation](index.md)
- Contrasts with: [Scientific Evidence](scientific-evidence.md)
- See also: [Scientific Corrections](scientific-corrections.md)
- Depends on: [Static Analysis And CI](static-analysis-and-ci.md)
- See also: [Numerical Evaluation](../usage/numerical-evaluation.md)
