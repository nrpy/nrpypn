# Command Line and TwoPunctures

> Supply CLI inputs and interpret the printed TwoPunctures assignments. · Status: confirmed
> Up: [Usage](index.md)

## Summary

Run `python3 -m nrpypn.quasicirc` with the mass ratio, separation, and spin
components to print `P_t` and `P_r`. With missing inputs, the program instead
prints its orbital assumptions, TwoPunctures guidance, and argument help.
The mapping identifies object 1 with the plus puncture and object 2 with the
minus puncture. The program supplies text instructions, not a completed
TwoPunctures parameter file or a solver invocation.

## Detail

### Positional Inputs and Output

`quasicirc.main` defines float-valued positional arguments in the following
order. An equal-mass, separation-12, nonspinning request is shown below.

| Position | Argument | Meaning |
| --- | --- | --- |
| 1 | `qmassratio` | Mass ratio `m2/m1`, documented as `>= 1`. |
| 2 | `nr` | Total coordinate separation between the black holes. |
| 3, 4, 5 | `nchi1x`, `nchi1y`, `nchi1z` | Dimensionless spin components of object 1. |
| 6, 7, 8 | `nchi2x`, `nchi2y`, `nchi2z` | Dimensionless spin components of object 2. |

```bash
python3 -m nrpypn.quasicirc 1 12 0 0 0 0 0 0
```

Claim evidence:
- Claim: The module command accepts the listed float-valued positional inputs in order; the example selects equal masses, separation `12`, and zero spins.
- Role: descriptive behavior
- Deciding authority: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, ordered `parser.add_argument` calls and the `__main__` block; [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval` mass-ratio definition.
- Corroboration: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`, accepts the same physical inputs in the forwarded order.

With all positional inputs present, `main` calls
`eval__P_t__and__P_r` and prints the returned values using
`print(f"P_t: {pt}, P_r: {pr}")`. It does not take absolute values in this
branch. The parser checks conversion to `float`, and the program checks
whether values are missing; neither that branch nor the numerical function
enforces the documented mass-ratio bound or physical spin and separation
ranges.

Claim evidence:
- Claim: Complete CLI input produces the evaluator's signed results in the `P_t: ..., P_r: ...` format without applying absolute values or enforcing physical input ranges.
- Role: descriptive behavior
- Deciding authority: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, complete-input branch; [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r` body.
- Corroboration: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval`, performs substitutions without physical-range checks.

### Missing Inputs and Help

Each positional argument has `nargs="?"`. After parsing, if any argument is
`None`, `main` prints the orbital assumptions and TwoPunctures instructions,
then calls `parser.print_help()`. It does not evaluate momenta in that branch.
Thus both an invocation with no positional inputs and one with only some
inputs take the guidance branch. These are not implicit default physical
parameters. `--help` requests argparse's argument help; the extra
TwoPunctures guidance is written in the separate missing-input branch.

Claim evidence:
- Claim: Missing any positional input selects printed guidance and help without momentum evaluation; `nargs="?"` does not supply default physical parameters, and the TwoPunctures guidance belongs to this branch rather than the `--help` text.
- Role: descriptive behavior
- Deciding authority: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, `nargs="?"` declarations, `any(arg is None for arg in vars(args).values())`, and the two resulting branches.
- Corroboration: [README.md](../../README.md), `### Quick Install from the Command Line (Bash Shell)`, documents invocation without positional arguments to obtain instructions.

### Orbital Geometry and Object Labels

The printed assumptions place object 1 on the positive x-axis and object 2
on the negative x-axis, with the center of mass at the origin and the
instantaneous orbit in the x–y plane. The implementation's Hamiltonian
substitutions use `nU = (1, 0, 0)`,
`p1U = (-Pr, Pt, 0)`, and `p2U = (Pr, -Pt, 0)`.

Claim evidence:
- Claim: NRPyPN's printed geometry places object 1 at positive x and object 2 at negative x in an instantaneous x–y orbit centered at the origin; the planar Hamiltonian uses opposite momenta `(-Pr, Pt, 0)` and `(Pr, -Pt, 0)` and `nU = (1, 0, 0)`.
- Role: descriptive behavior
- Deciding authority: [quasicirc.py](../../nrpypn/quasicirc.py), `main` missing-input orbital-assumption messages; [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary.make_replacements`.
- Corroboration: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), derived-vector declarations, sets `p1U = pU`, `p2U = -pU`, `n12U = nU`, and `n21U = -nU`.

The no-input instructions associate the plus puncture with object 1 and
the minus puncture with object 2. With `q = qmassratio = m2/m1`, the
following assignments are printed. In particular, for `q > 1` the minus
target mass is larger. A plus label does not mean the larger mass.

| TwoPunctures parameter | Printed assignment |
| --- | --- |
| `TwoPunctures::give_bare_mass` | `no` |
| `TwoPunctures::target_m_minus` | `q/(1+q)` |
| `TwoPunctures::target_m_plus` | `1/(1+q)` |
| `TwoPunctures::par_P_plus[0]` | `-abs(P_r)` |
| `TwoPunctures::par_P_plus[1]` | `+abs(P_t)` |
| `TwoPunctures::par_P_minus[0]` | `+abs(P_r)` |
| `TwoPunctures::par_P_minus[1]` | `-abs(P_t)` |
| `TwoPunctures::par_S_plus[0]` | `nchi1x * target_m_plus**2` |
| `TwoPunctures::par_S_plus[1]` | `nchi1y * target_m_plus**2` |
| `TwoPunctures::par_S_plus[2]` | `nchi1z * target_m_plus**2` |

Claim evidence:
- Claim: The missing-input text prescribes `give_bare_mass = no`, minus and plus target masses `q/(1+q)` and `1/(1+q)`, opposite x/y momenta with the displayed absolute-value signs, and plus spins from object 1's dimensionless spin times the plus target mass squared.
- Role: descriptive behavior
- Deciding authority: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, literal `TwoPunctures::` messages in the missing-input branch; absolute-value bars in those messages are written as `abs(...)` in the table.
- Corroboration: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval`, uses the same object-mass normalization and `S_a = chi_a * m_a**2`; [PN_p_r.py](../../nrpypn/PN_p_r.py), `f_Htot_xyplane_binary.make_replacements`, supports the planar component signs.

For the minus spin the program prints only “similarly for
`TwoPunctures::par_S_minus[]`.” Expanding that instruction using the
implemented object-2 spin substitution gives
`par_S_minus[i] = nchi2[i] * target_m_minus**2` for x, y, and z. Here
`nchi2[i]` means `nchi2x`, `nchi2y`, or `nchi2z`, respectively; this
componentwise expression is an expansion of the abbreviated guidance, not
an additional literal output line.

Claim evidence:
- Claim: The source abbreviates the minus-spin instruction; its componentwise expansion uses object 2's dimensionless spin and the square of the minus target mass, consistent with `num_eval`'s object-2 substitution.
- Role: descriptive behavior
- Deciding authority: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, the final minus-spin guidance message; [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval`, definitions of `nS2U0`, `nS2U1`, and `nS2U2`.
- Corroboration: none available as an independent TwoPunctures integration test; the source establishes the printed instruction and its numerical mass/spin convention.

### Limits of the Printed Mapping

The guidance applies absolute values when describing the x/y momentum
assignments; the numerical-output branch leaves the returned signs unchanged.
Neither branch writes these assignments to a parameter file or transfers
them to a solver. The guidance does not print z-momentum assignments or
puncture position/separation parameters, although the Hamiltonian
substitutions set both z momenta to zero. The printed list therefore does
not specify a complete TwoPunctures configuration or establish behavior of
an installed TwoPunctures solver.

Claim evidence:
- Claim: The printed mapping is text guidance with absolute-value x/y assignments, omits z-momentum and position/separation settings, and neither writes a complete parameter file nor invokes or validates a TwoPunctures solver; zero z momenta are established separately by the Hamiltonian substitutions.
- Role: descriptive behavior
- Deciding authority: [quasicirc.py](../../nrpypn/quasicirc.py), complete `main` body and its missing-input messages; [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary.make_replacements`.
- Corroboration: none available as a solver integration test; [README.md](../../README.md), `## Purpose`, presents TwoPunctures as an application of the computed momenta, without supplying solver execution evidence.

## Sources

- [quasicirc.py](../../nrpypn/quasicirc.py) — `main` positional arguments, missing-input messages, complete-input evaluator call and print, and the `__main__` block.
- [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r` input order and return statement.
- [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py) — `num_eval` mass and spin substitutions, and module-level derived-vector declarations.
- [PN_p_r.py](../../nrpypn/PN_p_r.py) — `PN_p_r.f_Htot_xyplane_binary.make_replacements`.
- [README.md](../../README.md) — `## Purpose` and `### Quick Install from the Command Line (Bash Shell)`.

## See Also

- Parent: [Usage](index.md)
- Depends on: [Numerical Evaluation](numerical-evaluation.md)
- Depends on: [Installation and Packaging](../architecture/installation-and-packaging.md)
- Depends on: [Symbols and Conventions](../equations/symbols-and-conventions.md)
- See also: [Calculation Overview](../architecture/overview.md)
- See also: [Scientific Evidence](../validation/scientific-evidence.md)
