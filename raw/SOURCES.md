# Source Manifest

> Registered source material for the NRPyPN knowledge base. Code and project
> documentation stay in place; imported KB conventions are preserved verbatim.

## Imported Conventions

| Source | Provenance | Status | Ingest |
| --- | --- | --- | --- |
| `raw/source-docs/kb-instructions.md` | Verbatim convention source from `nrpy-main/raw/source-docs/kb-instructions.md` in the reference NRPy repository; current local governance resolves superseded instructions. | frozen | ingested |
| `raw/source-docs/nrpy-kb-schema.md` | Verbatim convention source from `nrpy-main/wiki/SCHEMA.md` in the reference NRPy repository; current local governance resolves superseded instructions. | frozen | ingested |
| `raw/source-docs/nrpy-kb-workflows.md` | Verbatim convention source from `nrpy-main/wiki/workflows.md` in the reference NRPy repository; current local governance resolves superseded instructions. | frozen | ingested |
| `raw/source-docs/nrpy-kb-checks.md` | Verbatim convention source from `nrpy-main/wiki/lint/CHECKS.md` in the reference NRPy repository; current local governance resolves superseded instructions. | frozen | ingested |
| `raw/source-docs/nrpy-computational-physics-language.md` | Verbatim convention source from `nrpy-main/wiki/computational-physics-language.md` in the reference NRPy repository; current local governance resolves superseded instructions. | frozen | ingested |

## Dependency Evidence

| Source | Provenance | Status | Ingest |
| --- | --- | --- | --- |
| `raw/source-docs/nrpy-expression-validation.md` | Verbatim selected `compare_or_generate_trusted_results` and `output_trusted` definitions from NRPy's `nrpy/validate_expressions/validate_expressions.py` in the reference checkout. The preserved definitions establish the cited helper behavior, not behavior of every unpinned dependency release. | frozen | ingested |
| `raw/source-docs/nrpy-expression-sampling.md` | Verbatim selected `convert_free_symbols_set_to_mpf_dict` definition from NRPy's `nrpy/validate_expressions/validate_expressions.py` in the reference checkout. Describes the preserved sample-selection implementation, not every unpinned dependency release. | frozen | ingested |

## Repository Sources

Exact repository paths provide provenance. Source dependencies and ingest scope
are recorded in [Source Map](../wiki/source-map.md).

| Source | Status |
| --- | --- |
| `AGENTS.md` | living |
| `README.md` | living |
| `LICENSE` | living |
| `setup.py` | living |
| `requirements.txt` | living |
| `MANIFEST.in` | living |
| `.pylintrc` | living |
| `.pylintrc_python36` | living |
| `.github/workflows/main.yml` | living |
| `.github/standalone_bash_script.sh` | living |
| `tools/kb_lint.py` | living |
| `tools/test_kb_lint.py` | living |
| `.agents/README.md` | living |
| `.agents/review-protocol.md` | living |
| `.agents/skills/dialectic/SKILL.md` | living |
| `.agents/skills/trialectic/SKILL.md` | living |
| `nrpypn/NRPyPN_shortcuts.py` | living |
| `nrpypn/PN_Hamiltonian_NS.py` | living |
| `nrpypn/PN_Hamiltonian_SO.py` | living |
| `nrpypn/PN_Hamiltonian_SS.py` | living |
| `nrpypn/PN_Hamiltonian_SSS.py` | living |
| `nrpypn/PN_MOmega.py` | living |
| `nrpypn/PN_dE_GW_dt_and_dM_dt.py` | living |
| `nrpypn/PN_p_r.py` | living |
| `nrpypn/PN_p_t.py` | living |
| `nrpypn/__init__.py` | living |
| `nrpypn/eval_p_t_and_p_r.py` | living |
| `nrpypn/py.typed` | living |
| `nrpypn/quasicirc.py` | living |
| `nrpypn/release.txt` | living |
| `nrpypn/tests/PN_Hamiltonian_NS.py` | living |
| `nrpypn/tests/PN_Hamiltonian_SO.py` | living |
| `nrpypn/tests/PN_Hamiltonian_SS.py` | living |
| `nrpypn/tests/PN_Hamiltonian_SSS.py` | living |
| `nrpypn/tests/PN_MOmega.py` | living |
| `nrpypn/tests/PN_dE_GW_dt_and_dM_dt.py` | living |
| `nrpypn/tests/PN_p_r_AlternativeFalse.py` | living |
| `nrpypn/tests/PN_p_r_AlternativeTrue.py` | living |
| `nrpypn/tests/PN_p_t.py` | living |

## Exclusions

A nested `nrpy-main/` checkout is reference material, not an aggregate NRPyPN
source. Its selected convention documents are registered above as frozen imports.
Generated distributions, caches, bytecode, rendered outputs, images, scratch
reports, and maintenance logs remain outside the KB. Existing trusted expression
files stay in `nrpypn/tests/`; new generated values require justified scientific
review rather than automatic acceptance as evidence.
