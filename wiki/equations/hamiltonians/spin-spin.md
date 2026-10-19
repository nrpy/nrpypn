# Spin-Spin Hamiltonian

> Distinguish mixed-spin and self-spin expressions and their radial dependence. · Status: confirmed
> Up: [Hamiltonians](index.md)

## Summary

`PN_Hamiltonian_SS` separates its leading quadratic-spin expression from
mixed-spin and self-spin corrections. A mixed-spin term that vanishes on a
circular orbit can still contribute to radial momentum derivatives.

## Detail

### Stored quadratic-spin contributions

The constructor stores `H_SS_2PN`, `H_SS_S1S2_3PN`, and
`H_SS_S1sq_S2sq_3PN`. The last expression adds the per-body self-spin
contribution to the expression with body labels exchanged. Despite the
phrase “spin-orbit” in the docstring of `f_H_SS_S1sq_S2sq_3PN`, that
method's expression is quadratic in spin, including terms such as
`dot(p1U, S1U)**2` and `dot(S1U, S1U)`.

Claim evidence:
- Claim: PN_Hamiltonian_SS stores leading quadratic-spin, mixed-spin 3PN, and self-spin 3PN expressions; the self-spin method's spin-orbit wording conflicts with its quadratic-spin expression.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SS.py](../../../nrpypn/PN_Hamiltonian_SS.py), `PN_Hamiltonian_SS.__init__`, `f_H_SS_S1sq_S2sq_3PN` docstring, and its nested `f_H_SS_particle` expression.
- Corroboration: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary`, adds these attributes separately from the spin-orbit contributions. This supports their classification in the implementation, not derivation from a paper.

Claim status: contested; contradiction: CONTR-0004.
See [CONTR-0004](../../contradictions.md#contr-0004) for deciding evidence and resolution.

### Leading expression

With \(M=m_1+m_2\), \(\mu=m_1m_2/M\), and

\[
\mathbf S_0=\left(1+\frac{m_2}{m_1}\right)\mathbf S_1
+\left(1+\frac{m_1}{m_2}\right)\mathbf S_2,
\]

the code constructs

\[
H_{\rm SS,2PN}=\frac{\mu}{2M q^3}
\left[3(\mathbf S_0\cdot\mathbf n)^2-\mathbf S_0\cdot\mathbf S_0\right].
\]

Here `q` is the separation parameter of `f_H_SS_2PN`; the constructor
passes `r12` for it. The weighted-spin square includes both self-spin
and mixed-spin terms.

Claim evidence:
- Claim: f_H_SS_2PN implements the displayed weighted-spin expression with q as separation, including self-spin and mixed-spin terms.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SS.py](../../../nrpypn/PN_Hamiltonian_SS.py), `PN_Hamiltonian_SS.f_H_SS_2PN` and its call in `__init__`.
- Corroboration: [tests/PN_Hamiltonian_SS.py](../../../nrpypn/tests/PN_Hamiltonian_SS.py), `trusted_dict["H_SS_2PN"]`, provides a numerical expression reference only; it does not verify the displayed expression symbolically or derive it scientifically.

### Momentum choice and radial derivatives

In the block proportional to \(3/(2m_2^2r_{12}^3)\), the implemented
contribution is

\[
T=-\frac{3}{2m_2^2r_{12}^3}
(\mathbf S_2\cdot\mathbf n_{12})
(\mathbf S_1\cdot\mathbf p_2)
(\mathbf p_2\cdot\mathbf n_{12}).
\]

The momentum contracted with \(\mathbf S_1\) is \(\mathbf p_2\).
Under the radial module's center-of-mass substitutions,
\(\mathbf p_2=(P_r,-P_t,0)\) and \(\mathbf n_{12}=(1,0,0)\),
the factor \(\mathbf p_2\cdot\mathbf n_{12}=P_r\) makes this term
vanish at `Pr = 0`. Differentiation with respect to `Pr` can retain
contributions, including a quadratic term when the relevant spin components
are nonzero. Circular values alone therefore do not test its radial response.
The mixed-spin docstring reports a different momentum in earlier
transcriptions. References in docstrings are bibliographic attributions;
the implemented expression decides which momentum is used here.

Claim evidence:
- Claim: The indicated mixed-spin term uses S1 dot p2 and vanishes at zero radial momentum in the shortcut frame, while its radial momentum derivatives can survive; the docstring reports an earlier S1 dot p1 transcription.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SS.py](../../../nrpypn/PN_Hamiltonian_SS.py), `PN_Hamiltonian_SS.f_H_SS_S1S2_3PN`, the `3/(2*m2**2*r12**3)` block and its explanatory docstring.
- Corroboration: [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary.make_replacements` and `f_p_r_alternative__use_fullHam`, supplies the frame substitutions and takes first and second `Pr` derivatives; [README.md](../../../README.md), `Validation`, separately reports the momentum correction. Neither is an independent scientific derivation.

## Sources

- [PN_Hamiltonian_SS.py](../../../nrpypn/PN_Hamiltonian_SS.py) — `PN_Hamiltonian_SS.__init__`, `f_H_SS_2PN`, `f_H_SS_S1S2_3PN`, and `f_H_SS_S1sq_S2sq_3PN`.
- [PN_p_r.py](../../../nrpypn/PN_p_r.py) — `PN_p_r.f_Htot_xyplane_binary` and `f_p_r_alternative__use_fullHam`.
- [tests/PN_Hamiltonian_SS.py](../../../nrpypn/tests/PN_Hamiltonian_SS.py) — `trusted_dict`.
- [README.md](../../../README.md) — `Validation`.

## See Also

- Parent: [Hamiltonians](index.md)
- Depends on: [Symbols and conventions](../symbols-and-conventions.md)
- See also: [Cubic in spin](cubic-in-spin.md)
- See also: [Radial momentum](../orbital-quantities/radial-momentum.md)
- Validated by: [Expression regressions](../../validation/expression-regressions.md)
