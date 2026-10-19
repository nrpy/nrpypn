# Calculation Overview

> Locate symbolic expressions and follow their numerical evaluation. · Status: confirmed
> Up: [Architecture](index.md)

## Summary

NRPyPN constructs symbolic post-Newtonian expressions and evaluates tangential
and radial momenta for binary-black-hole initial data. The numerical function
substitutes masses, separation, and spins; the command-line module prints the
result. The README describes reduced eccentricity as a goal, rather than a
guaranteed property of an evolution.

## Detail

### Purpose and Calculation Boundary

The README identifies high-order momenta for initial data with minimal
eccentricity as the project's immediate goal. Its suggested application is
supplying momenta to an initial-data solver such as TwoPunctures. The
command-line implementation evaluates momenta and prints text; it does not
invoke TwoPunctures or evolve a binary. These sources establish purpose and
implemented operations, not a measured eccentricity guarantee.

Claim evidence:
- Claim: The documented goal is momenta for low-eccentricity initial data; the CLI evaluates and prints momenta without invoking an initial-data solver or guaranteeing an evolution's eccentricity.
- Role: descriptive behavior
- Deciding authority: [README.md](../../README.md), `## Purpose`; [quasicirc.py](../../nrpypn/quasicirc.py), `main`.
- Corroboration: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`, returns numerical momenta without a solver call.

### Symbolic Expressions

The equation classes construct expressions on their instances. The following
names identify stored expressions, not independently established accuracy
guarantees.

| Defining module or class | Expressions used in the calculation |
| --- | --- |
| `NRPyPN_shortcuts` | Shared masses, separation symbols, spin and momentum vectors, vector operations, and `num_eval` substitutions. |
| `PN_Hamiltonian_NS` | `H_Newt`, `H_NS_1PN`, `H_NS_2PN`, `H_NS_3PN`. |
| `PN_Hamiltonian_SO` | `H_SO_1p5PN`, `H_SO_2p5PN`, `H_SO_3p5PN`. |
| `PN_Hamiltonian_SS` | `H_SS_2PN`, `H_SS_S1S2_3PN`, `H_SS_S1sq_S2sq_3PN`. |
| `PN_Hamiltonian_SSS` | `H_SSS_3PN`; retain the identifier when looking up the expression. |
| `PN_p_t` | `p_t` and its coefficient attributes `a_2` through `a_7`. |
| `PN_MOmega` | `MOmega` and its coefficient attributes `a_2` through `a_7`. |
| `PN_dE_GW_dt_and_dM_dt` | `dE_GW_dt`, `dM_dt`, and `dE_GW_dt_plus_dM_dt`. |
| `PN_p_r` | `Htot_xyplane_binary`, `dr_dt`, and `p_r`. |

Claim evidence:
- Claim: The listed modules and classes define the shared symbols or store the named symbolic expressions; their identifiers alone do not establish scientific accuracy.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), module-level declarations and `num_eval`; the `__init__` methods and expression-building methods of [PN_Hamiltonian_NS](../../nrpypn/PN_Hamiltonian_NS.py), [PN_Hamiltonian_SO](../../nrpypn/PN_Hamiltonian_SO.py), [PN_Hamiltonian_SS](../../nrpypn/PN_Hamiltonian_SS.py), [PN_Hamiltonian_SSS](../../nrpypn/PN_Hamiltonian_SSS.py), [PN_p_t](../../nrpypn/PN_p_t.py), [PN_MOmega](../../nrpypn/PN_MOmega.py), [PN_dE_GW_dt_and_dM_dt](../../nrpypn/PN_dE_GW_dt_and_dM_dt.py), and [PN_p_r](../../nrpypn/PN_p_r.py).
- Corroboration: [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary` and `PN_p_r.f_dr_dt`, access these expression attributes in the radial-momentum calculation.

### Numerical Call Sequence

`eval__P_t__and__P_r` constructs `PN_p_t` and `PN_p_r` using the shared
symbols. It evaluates `pt.p_t` with `num_eval`, evaluates `pr.p_r` with the
same physical inputs plus the computed `nPt`, and returns
`(float(nPt), float(nPr))`. It constructs the expressions on each call; there
is no saved-expression cache in this function.

Claim evidence:
- Claim: `eval__P_t__and__P_r` constructs symbolic momenta on each call, substitutes the tangential result into the radial evaluation, and explicitly converts the returned pair to Python floats.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`.
- Corroboration: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, calls this function and unpacks its pair as `pt, pr`.

The radial expression depends on the Hamiltonian modules as well as the
frequency and flux modules. `f_Htot_xyplane_binary` sums Hamiltonian terms
after imposing the instantaneous orbital geometry. `f_dr_dt` constructs a
`PN_p_t`, differentiates the circular Hamiltonian along that expression,
constructs `PN_MOmega`, and passes its `MOmega` into
`PN_dE_GW_dt_and_dM_dt`. It assigns the ratio of the combined flux expression
to that Hamiltonian derivative to `self.dr_dt`.

Claim evidence:
- Claim: The implemented radial calculation combines the Hamiltonian expressions, a tangential-momentum expression, orbital frequency, and the combined flux through `f_Htot_xyplane_binary` and `f_dr_dt`; this call sequence is a description of code, not a derivation of the equations.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary` and `PN_p_r.f_dr_dt`.
- Corroboration: [PN_dE_GW_dt_and_dM_dt.py](../../nrpypn/PN_dE_GW_dt_and_dM_dt.py), `PN_dE_GW_dt_and_dM_dt.__init__` and `f_dE_GW_dt_and_dM_dt`, define the combined flux attribute accessed by `f_dr_dt`.

### Evaluation and Validation Are Different Entry Points

`python3 -m nrpypn.quasicirc` enters `quasicirc.main`. By contrast, the
`__main__` blocks of `PN_p_t.py` and `PN_p_r.py` run doctests and call
`ve.process_dictionary_of_expressions` followed by
`ve.compare_or_generate_trusted_results`. Their latter calls pass
`os.getcwd()` as a path argument. Treat these as expression-validation
programs with output-related calls, rather than interchangeable momentum
printing commands.

Claim evidence:
- Claim: The CLI enters `quasicirc.main`, while the tangential and radial equation scripts enter doctest and expression-comparison or generation code that receives the working-directory path.
- Role: descriptive behavior
- Deciding authority: [quasicirc.py](../../nrpypn/quasicirc.py), `if __name__ == "__main__"`; [PN_p_t.py](../../nrpypn/PN_p_t.py) and [PN_p_r.py](../../nrpypn/PN_p_r.py), their `if __name__ == "__main__"` blocks.
- Corroboration: [README.md](../../README.md), `### Quick Install from the Command Line (Bash Shell)`, selects `nrpypn.quasicirc` as the user-facing command.

The README's `Validation` section explicitly distinguishes agreement with a
printed reference from correctness of that reference. The same limit applies
when using numerical regressions to assess changes: agreement with stored
values does not establish a scientific derivation or independence of the
underlying references.

Claim evidence:
- Claim: Reference-transcription agreement and stored-value regressions do not independently prove scientific correctness or derivation.
- Role: public/scientific contract
- Deciding authority: [README.md](../../README.md), `## Validation`.
- Corroboration: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `test_results_against_trusted`, compares numerical momenta with stored values; it supplies no separate scientific derivation.

## Sources

- [README.md](../../README.md) — `## Purpose`, `## Validation`, and `### Quick Install from the Command Line (Bash Shell)`.
- [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py) — shared declarations and `num_eval`.
- [PN_Hamiltonian_NS.py](../../nrpypn/PN_Hamiltonian_NS.py) — `PN_Hamiltonian_NS.__init__`, `f_H_Newt__H_NS_1PN__H_NS_2PN`, and `f_H_NS_3PN`.
- [PN_Hamiltonian_SO.py](../../nrpypn/PN_Hamiltonian_SO.py) — `PN_Hamiltonian_SO.__init__` and `f_H_SO_1p5PN`, `f_H_SO_2p5PN`, `f_H_SO_3p5PN`.
- [PN_Hamiltonian_SS.py](../../nrpypn/PN_Hamiltonian_SS.py) — `PN_Hamiltonian_SS.__init__` and `f_H_SS_2PN`, `f_H_SS_S1S2_3PN`, `f_H_SS_S1sq_S2sq_3PN`.
- [PN_Hamiltonian_SSS.py](../../nrpypn/PN_Hamiltonian_SSS.py) — `PN_Hamiltonian_SSS.__init__` and `f_H_SSS_3PN`.
- [PN_p_t.py](../../nrpypn/PN_p_t.py) — `PN_p_t.__init__`, `f_p_t`, and the `__main__` block.
- [PN_MOmega.py](../../nrpypn/PN_MOmega.py) — `PN_MOmega.__init__` and `f_MOmega`.
- [PN_dE_GW_dt_and_dM_dt.py](../../nrpypn/PN_dE_GW_dt_and_dM_dt.py) — `PN_dE_GW_dt_and_dM_dt.__init__` and `f_dE_GW_dt_and_dM_dt`.
- [PN_p_r.py](../../nrpypn/PN_p_r.py) — `PN_p_r.__init__`, `f_Htot_xyplane_binary`, `f_dr_dt`, and the `__main__` block.
- [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r` and `test_results_against_trusted`.
- [quasicirc.py](../../nrpypn/quasicirc.py) — `main` and the `__main__` block.

## See Also

- Parent: [Architecture](index.md)
- Depends on: [Installation and Packaging](installation-and-packaging.md)
- Example: [Numerical Evaluation](../usage/numerical-evaluation.md)
- See also: [Command Line and TwoPunctures](../usage/command-line-and-twopunctures.md)
- Depends on: [Symbols and Conventions](../equations/symbols-and-conventions.md)
- Validated by: [Expression Regressions](../validation/expression-regressions.md)
- See also: [Scientific Evidence](../validation/scientific-evidence.md)
