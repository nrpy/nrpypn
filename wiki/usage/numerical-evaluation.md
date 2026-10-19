# Numerical Evaluation

> Evaluate tangential and radial momenta with the Python API. · Status: confirmed
> Up: [Usage](index.md)

## Summary

Import `eval__P_t__and__P_r` from `nrpypn.eval_p_t_and_p_r` and supply the
mass ratio, coordinate separation, and dimensionless spin components. It
returns `(P_t, P_r)` as Python floats. The evaluator normalizes the total mass
to one and uses the default closed-form radial calculation. Physical input
conditions are documented assumptions, rather than enforced range checks.

## Detail

### Calling the Numerical Function

The function takes the following arguments, in this order, with no defaults:

| Argument | Meaning |
| --- | --- |
| `qmassratio` | Mass ratio `m2/m1`, documented as `>= 1`. |
| `nr` | Total coordinate distance between the black holes. |
| `nchi1x`, `nchi1y`, `nchi1z` | Dimensionless spin components of black hole 1. |
| `nchi2x`, `nchi2y`, `nchi2z` | Dimensionless spin components of black hole 2. |

Claim evidence:
- Claim: `eval__P_t__and__P_r` requires the listed mass ratio, coordinate separation, and dimensionless spin components without defaults, using `qmassratio = m2/m1` and the documented assumption `qmassratio >= 1`.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r` signature and parameter docstring; [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval` mass substitutions.
- Corroboration: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, defines and forwards the corresponding positional inputs in the same order.

For example, this call requests equal masses, separation `12`, and zero
spins. The return statement explicitly converts both computed values to
Python `float` objects and orders the tangential result before the radial
result.

```python
from nrpypn.eval_p_t_and_p_r import eval__P_t__and__P_r

p_t, p_r = eval__P_t__and__P_r(
    qmassratio=1.0,
    nr=12.0,
    nchi1x=0.0,
    nchi1y=0.0,
    nchi1z=0.0,
    nchi2x=0.0,
    nchi2y=0.0,
    nchi2z=0.0,
)
```

Claim evidence:
- Claim: The example selects equal masses, separation `12`, and zero spins; the evaluator returns `(float(nPt), float(nPr))` in tangential-then-radial order.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`, its `num_eval` calls and return statement.
- Corroboration: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, forwards these inputs and unpacks the return pair as `pt, pr`.

### Mass, Spin, and Separation Substitutions

For `Q = qmassratio`, `num_eval` sets

\[
m_1=\frac{1}{1+Q},\qquad m_2=\frac{Q}{1+Q},\qquad m_1+m_2=1,
\]

and substitutes spin angular momentum components as
`S1U[0] = nchi1x * m1**2` and `S2U[0] = nchi2x * m2**2`, with the
corresponding y and z inputs used for components 1 and 2. It also
substitutes the dimensionless `chi1U` and `chi2U` components directly.
Thus black hole 2 has the larger mass when `Q > 1`; the input spins must
retain their object labels.

Claim evidence:
- Claim: `num_eval` normalizes `m1+m2` to one with `m2/m1 = qmassratio` and substitutes `S_a = chi_a * m_a**2` componentwise; for a mass ratio above one, object 2 is the more massive object.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval`, definitions of `nm1`, `nm2`, `nS1U0` through `nS2U2`, and the substitution chain.
- Corroboration: [quasicirc.py](../../nrpypn/quasicirc.py), `main` missing-input branch, associates `target_m_plus` with `1/(1+q)`, `target_m_minus` with `q/(1+q)`, and the first spin with the plus mass squared.

The module-level symbol `q` in `NRPyPN_shortcuts` can represent a
separation, whereas `qmassratio` is the numerical mass ratio. `num_eval`
substitutes `nr` for both shared separation symbols `r` and `q`. It also
replaces `pi` and the Euler–Mascheroni constant with SymPy numerical values.
The mass ratio is supplied through `nm1` and `nm2`, not by substituting the
shared symbol `q` with `qmassratio`.

Claim evidence:
- Claim: `num_eval` substitutes `nr` for both shared symbols `r` and `q`, supplies the mass ratio through the mass substitutions, and numerically substitutes `pi` and the Euler–Mascheroni constant.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), declarations of `r`, `q`, and `gamma_EulerMascheroni`, and `num_eval`.
- Corroboration: [PN_p_t.py](../../nrpypn/PN_p_t.py), `PN_p_t.f_p_t`, constructs its local ratio as `m2/m1` while using its separate `r` argument for separation.

### Tangential Evaluation and the Radial Method

The wrapper constructs `PN_p_t` and `PN_p_r`, evaluates `pt.p_t`, then
evaluates `pr.p_r` with that tangential value passed as `nPt`. It does not
offer a radial-method argument. `PN_p_r.__init__` defaults
`use_alternative_approach_for_p_r` to `False`, which selects `f_p_r`.
Choosing `True` requires constructing `PN_p_r` directly and selects
`f_p_r_alternative__use_fullHam`. The alternative uses Hamiltonian
derivatives without the same self-consistent truncation of higher-order
terms; the two methods should not be treated as exactly equal expressions.

Claim evidence:
- Claim: The numerical wrapper evaluates the default closed-form radial expression using the evaluated tangential momentum; selecting the alternative full-Hamiltonian method requires direct `PN_p_r` construction, and the source does not promise exact equality between the methods.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`; [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.__init__`, class docstring, `f_p_r`, and `f_p_r_alternative__use_fullHam`.
- Corroboration: none available as an independent method-equivalence test; the `PN_p_r.py` `__main__` block compares each alternative with its own trusted-expression data rather than asserting equality of the alternatives.

References in docstrings are bibliographic attributions; this page describes
the implementation rather than independently verifying its derivation. See
[Scientific Evidence](../validation/scientific-evidence.md) for the distinction
between implementation comparisons and scientific support.

### Lower-Level Substitution Limits

`num_eval` defaults to mass ratio `1.0`, separation `12.0`, zero spins,
and `nPt=None`, `ndrdt=None`. Its optional substitutions use `if nPt:` and
`if ndrdt:`. Consequently, passing `0.0` skips the corresponding substitution
just as `None` does. If the expression contains the corresponding free
symbol, that symbol remains unless the caller substitutes it separately.

Claim evidence:
- Claim: `num_eval` has the listed defaults and performs optional `Pt` and `drdt` substitutions only for truthy values; explicit zero therefore skips those substitutions.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval` signature and its `if nPt` and `if ndrdt` branches.
- Corroboration: none available; deterministic inspection of these branches decides zero handling, and `test_results_against_trusted` does not target an expression requiring an optional zero substitution.

Although annotated to return `float`, `num_eval` returns a
`typing.cast(float, substituted_expression)`. That cast is not an explicit
numeric conversion. An expression with remaining free symbols can therefore
remain symbolic. The outer momentum evaluator performs actual `float(...)`
conversions, so the lower-level helper's annotation must not be read as a
guarantee that arbitrary expressions become Python floats.

Claim evidence:
- Claim: `num_eval` returns the substituted expression through `typing.cast`, whereas `eval__P_t__and__P_r` explicitly applies `float`; the helper does not guarantee that arbitrary free symbols are eliminated or that its returned object is a Python float.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval` return expression and `cast` import; [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r` return statement.
- Corroboration: none available; the registered numerical regression function checks momentum values, not the return type of `num_eval` on arbitrary symbolic inputs.

Claim status: contested; contradiction: CONTR-0002.
See [CONTR-0002](../contradictions.md#contr-0002) for deciding evidence and resolution.

Neither the numerical wrapper nor `num_eval` enforces the documented
`qmassratio >= 1` assumption or checks positive separation, finite inputs,
or spin magnitude. They also expose no precision or error-tolerance argument.
The caller is responsible for supplying inputs appropriate to the model;
successful substitution alone does not establish physical validity or an
error bound.

Claim evidence:
- Claim: The numerical wrapper and substitution helper perform no physical-range or finiteness checks and expose no precision or tolerance controls; their documented input assumptions are caller responsibilities rather than validated conditions.
- Role: descriptive behavior
- Deciding authority: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), complete `eval__P_t__and__P_r` signature and body; [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), complete `num_eval` signature and body.
- Corroboration: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, converts positional values with `type=float` and checks for missing values without adding physical-range checks.

## Sources

- [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r` and `test_results_against_trusted`.
- [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py) — shared symbols and `num_eval`.
- [PN_p_t.py](../../nrpypn/PN_p_t.py) — `PN_p_t.__init__` and `f_p_t`.
- [PN_p_r.py](../../nrpypn/PN_p_r.py) — `PN_p_r.__init__`, class docstring, `f_p_r`, `f_p_r_alternative__use_fullHam`, and the `__main__` block.
- [quasicirc.py](../../nrpypn/quasicirc.py) — `main` argument definitions, evaluator call, and missing-input guidance.

## See Also

- Parent: [Usage](index.md)
- Depends on: [Installation and Packaging](../architecture/installation-and-packaging.md)
- Depends on: [Symbols and Conventions](../equations/symbols-and-conventions.md)
- See also: [Calculation Overview](../architecture/overview.md)
- Example: [Command Line and TwoPunctures](command-line-and-twopunctures.md)
- Validated by: [Expression Regressions](../validation/expression-regressions.md)
- See also: [Scientific Evidence](../validation/scientific-evidence.md)
