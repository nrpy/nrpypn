# Static Analysis And CI

> Apply Python quality rules and distinguish configured checks from execution evidence. · Status: contested
> Up: [Validation](index.md)

## Summary

New handwritten Python files require Pylint `10.00/10.00`; existing handwritten
files must not regress from their pre-change score. The configured workflow and
standalone script instead compare parsed Pylint scores with `9.5`. They also
execute modules and perform formatting and type checks with a legacy-Python
exception. Configuration describes intended checks, not successful runs.
Use scoped commands in an isolated copy: the standalone script deletes the
NRPy cache, installs dependencies, and recursively discovers Python files.

## Detail

### Adopted Python Quality Requirements

The root contributor rule requires Pylint `10.00/10.00` for newly added
handwritten Python. Existing handwritten files retain their pre-change score
without regression, including a grandfathered score at or below `9.5`.
Compare the existing and proposed files with the same Pylint configuration and
tool version. Generated trusted dictionaries are exempt from per-file
analysis; their handwritten calculation and comparison code is not. Do not
add suppressions or weaken scientific checks to obtain a passing result.

Claim evidence:
- Claim: New handwritten Python requires Pylint `10.00/10.00`; existing handwritten Python must not regress from its pre-change score, and generated trusted dictionaries alone receive the per-file exemption.
- Role: normative rule
- Deciding authority: [AGENTS.md](../../AGENTS.md), `## Python Quality Requirements`
- Corroboration: [agent guide](../../.agents/README.md), `## Validation`, handwritten-Python checks and score requirements

### Configured Workflow

The workflow's `on.push.branches` selects `main`. Its `jobs.build` commands
install dependencies and discover Python files recursively from the checkout
root, excluding `__init__.py`, `build/`, `*/tests/*`, and the named CarpetX
path. For discovered files it executes the module with `DOCTEST_MODE=1`, then
checks Black and strict mypy except in the legacy matrix branch, and parses a
Pylint score. A separate final step calls
`eval_p_t_and_p_r.test_results_against_trusted()`. These configured commands
do not establish that a run passed or that a formula is scientifically valid.

Claim evidence:
- Claim: The workflow configures module execution, conditional Black and mypy, Pylint score parsing, and a separate momentum regression on pushes to `main`; its text is not evidence of a successful execution.
- Role: CI behavior
- Deciding authority: [main.yml](../../.github/workflows/main.yml), `on.push.branches`, `jobs.build`, steps `Run doctests, black, mypy, and pylint on each file.` and `Run validation tests, comparing p_t and p_r against trusted values from original implementation of NRPyPN.`
- Corroboration: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`, the configured final comparison function

Both Pylint configuration files set `[MASTER]` `fail-under=10`. The workflow
and standalone script extract the displayed score, test whether it is below
`9.5`, and suppress nonzero exits from their diagnostic Pylint calls with
`|| true`. They neither distinguish new handwritten files from existing ones
nor compare an existing file's score before and after a change. Thus their
configured score check does not enforce the adopted contributor rule; a
reviewer must apply that rule separately. A new file can meet the configured
threshold without meeting `10.00`, and an existing file's lower score can
remain above `9.5`. Conversely, the configured threshold can reject a
grandfathered existing score below `9.5` even when it has not regressed.

Claim evidence:
- Claim: The workflow and standalone script use a parsed-score threshold of `9.5`, despite `fail-under=10` in the Pylint configurations, and do not enforce the new-file or no-regression distinction.
- Role: CI behavior
- Deciding authority: [main.yml](../../.github/workflows/main.yml), `jobs.build` inline `pylint_score` comparison; [standalone_bash_script.sh](../../.github/standalone_bash_script.sh), `pylint_score` comparison and diagnostic calls
- Corroboration: [.pylintrc](../../.pylintrc) and [.pylintrc_python36](../../.pylintrc_python36), `[MASTER]` `fail-under`; [AGENTS.md](../../AGENTS.md), `## Python Quality Requirements`

Claim status: contested; contradiction: CONTR-0001.
See [CONTR-0001](../contradictions.md#contr-0001) for the unresolved difference
between contributor requirements and configured score enforcement.

The final workflow step calls its references values from the original
implementation. The deciding test-source comments state that spinning cases
were regenerated after scientific corrections. Read that label as historical
wording, not a guarantee that the values remain unchanged from the original
implementation.

Claim evidence:
- Claim: The workflow's original-implementation label does not guarantee unchanged reference values; the test documents regenerated spinning-case values.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`, comments preceding `case_dict`
- Corroboration: [README.md](../../README.md), `## Validation`; competing wording is the final validation step name in [main.yml](../../.github/workflows/main.yml)

Claim status: contested; contradiction: CONTR-0006.
See [CONTR-0006](../contradictions.md#contr-0006) for the reference-provenance
qualification and the resolution test for the workflow label.

### Scoped Local Checks

NRPyPN does not provide the upstream single-file wrapper. Apply equivalent
scoped checks from the NRPyPN root in an isolated intended-change copy with
owned output and cache directories. Inspect the selected module's `__main__`
effects and require its expected trusted files before execution. The following
example selects a handwritten equation module:

```bash
changed_file=nrpypn/PN_p_t.py
DOCTEST_MODE=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python "$changed_file"
python -m black --check "$changed_file"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python -m mypy --strict --pretty --allow-untyped-calls "$changed_file"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python -m pylint --rcfile=.pylintrc "$changed_file"
```

Claim evidence:
- Claim: Scoped module execution, Black, strict mypy with `--allow-untyped-calls`, and Pylint are the local check route when the upstream wrapper is absent; the adopted Pylint score requirements still apply.
- Role: normative rule
- Deciding authority: [AGENTS.md](../../AGENTS.md), `## Python Quality Requirements`
- Corroboration: [agent guide](../../.agents/README.md), `## Validation`, scoped command example

Use `.pylintrc_python36` for the declared legacy interpreter; the configured
Python `3.6.7` branch skips Black and mypy. Preserve declared compatibility and
report unavailable checks as missing checks. Pylint's direct exit status can
reflect `fail-under=10`, so judge a grandfathered existing file using the
required score comparison and its diagnostics rather than treating the
workflow's `9.5` threshold as permission to regress. When momenta or numerical
substitutions change, also run the physical-case regression with the
[finite-output wrapper](expression-regressions.md#physical-case-momentum-regression).
The unwrapped comparison can accept NaNs, so its exit status alone does not
establish finite numerical agreement.

Claim evidence:
- Claim: The legacy workflow branch omits Black and mypy and uses `.pylintrc_python36`; changes to momenta or substitutions additionally require finite evaluated momenta and the stored physical-case comparison.
- Role: normative rule
- Deciding authority: [agent guide](../../.agents/README.md), `## Validation`
- Corroboration: [main.yml](../../.github/workflows/main.yml), `matrix.python-version` conditions; [.pylintrc_python36](../../.pylintrc_python36), `[MASTER]` `fail-under`

### Standalone Script Effects And Isolation

The standalone script starts with `rm -rf ~/.cache/nrpy`, upgrades pip,
installs dependencies and analysis tools, and builds `python_files` with
`find .`. Its exclusions do not remove an entire nested `nrpy-main/` checkout,
so running it from a shared root can execute unrelated reference Python.
With the preserved delegated helper, `DOCTEST_MODE=1` also does not prevent
missing trusted dictionaries from being written. Inspect the imported helper
on dependency changes, as described in [Expression
Regressions](expression-regressions.md#missing-data-and-radial-momentum-branches).
An isolated checkout alone does not isolate the user's home cache.

Claim evidence:
- Claim: The standalone script deletes `~/.cache/nrpy`, installs packages, and recursively executes discovered Python without excluding a nested `nrpy-main/`; with the preserved delegated helper, module execution can also generate missing trusted data.
- Role: descriptive behavior
- Deciding authority: [standalone_bash_script.sh](../../.github/standalone_bash_script.sh), opening cache removal, dependency installation, `python_files`, and module-execution loop; [preserved NRPy helper definitions](../../raw/source-docs/nrpy-expression-validation.md), `## compare_or_generate_trusted_results` and `## output_trusted`
- Corroboration: [agent guide](../../.agents/README.md), `## Validation`, standalone-script and `DOCTEST_MODE` warnings

Prefer the scoped commands above. If reproducing the standalone script is
separately authorized, use an owned disposable copy containing only the
intended NRPyPN tree, an isolated home and cache, owned outputs, and bounded
execution resources. Inspect installation and network effects first. Never
delete a shared cache or execute the recursive script in the shared tree.
Workflow edits require explicit authorization naming
`.github/workflows/main.yml`.

Claim evidence:
- Claim: Commands with output or deletion effects require owned isolation; the recursive standalone script must not run blindly in the shared tree, and the protected workflow requires exact-file authorization to edit.
- Role: normative rule
- Deciding authority: [AGENTS.md](../../AGENTS.md), `## Python Quality Requirements` and `## Protected Workflow File`
- Corroboration: [agent guide](../../.agents/README.md), `## Validation`; [standalone_bash_script.sh](../../.github/standalone_bash_script.sh), opening cache removal and recursive discovery

## Sources

- [AGENTS.md](../../AGENTS.md) — `## Python Quality Requirements`, `## Protected Workflow File`.
- [Agent guide](../../.agents/README.md) — `## Validation`, `## Scientific Standards`.
- [CI workflow](../../.github/workflows/main.yml) — `on.push.branches`, `jobs.build`, per-file loop and final validation step.
- [Standalone script](../../.github/standalone_bash_script.sh) — cache removal, installation, `python_files`, module loop, score comparison.
- [Preserved NRPy helper definitions](../../raw/source-docs/nrpy-expression-validation.md) — `## compare_or_generate_trusted_results`, `## output_trusted`; selected dependency behavior only.
- [Pylint configuration](../../.pylintrc) — `[MASTER]` `fail-under`.
- [Legacy Pylint configuration](../../.pylintrc_python36) — `[MASTER]` `fail-under`.
- [Numerical evaluator](../../nrpypn/eval_p_t_and_p_r.py) — `test_results_against_trusted`, comments preceding `case_dict`.
- [README.md](../../README.md) — `## Validation`.

## See Also

- Parent: [Validation](index.md)
- Depends on: [Expression Regressions](expression-regressions.md)
- Contrasts with: [Scientific Evidence](scientific-evidence.md)
- See also: [Scientific Corrections](scientific-corrections.md)
- See also: [Architecture Overview](../architecture/overview.md)
