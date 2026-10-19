# Cubic-in-Spin Hamiltonian

> Interpret the SSS name and trace the implemented coefficient into orbital quantities. · Status: confirmed
> Up: [Hamiltonians](index.md)

## Summary

`PN_Hamiltonian_SSS` implements the leading cubic-spin expression under
the retained attribute name `H_SSS_3PN`. Its docstrings distinguish that name
from the rapidly rotating-body PN classification and explain a coefficient
change shared by related orbital expressions.

## Detail

### Identifier versus PN classification

The constructor calls `f_H_SSS_3PN` and stores `H_SSS_3PN`. The module,
class, and method docstrings describe this as the leading cubic-in-spin term,
3.5PN for rapidly rotating bodies, while retaining the tutorial's `3PN`
label in identifiers. The suffix alone should not determine PN ordering.
This classification is the repository's stated interpretation; references
in its docstrings are bibliographic attributions rather than independent
verification of the derivation.

Claim evidence:
- Claim: H_SSS_3PN is the retained identifier for the leading cubic-spin expression, which the defining docstrings classify as 3.5PN for rapidly rotating bodies.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SSS.py](../../../nrpypn/PN_Hamiltonian_SSS.py), module docstring, `PN_Hamiltonian_SSS` class docstring, `__init__`, and `f_H_SSS_3PN`.
- Corroboration: none available for an independent scientific PN classification; the repeated docstrings belong to the same defining source and are not independent derivations.

### Coefficient and body exchange

The nested `f_H_SSS_3PN_pt` builds one contribution; `f_H_SSS_3PN` adds
the contribution with body labels exchanged. In the first contribution, the
block with an explicit `m1/m2` factor is

\[
-\frac{9m_1}{4m_2}\frac{1}{m_1^2r^4}
\left[\mathbf S_1^2\,\mathbf S_2\cdot(\mathbf n\times\mathbf p_2)
+2(\mathbf S_1\cdot\mathbf n)\,
\mathbf S_2\cdot(\mathbf S_1\times\mathbf p_2)
-5(\mathbf S_1\cdot\mathbf n)^2\,
\mathbf S_2\cdot(\mathbf n\times\mathbf p_2)\right].
\]

Thus its coefficient before the common denominator is `-9*m1/(4*m2)`.
The class docstring reports `-3/2` instead of `-9/4` in earlier
transcriptions of that factor. Code establishes which expression is used
here; the docstring's account of the cited equations does not itself
establish which printed derivation is correct.

Claim evidence:
- Claim: The implemented per-body m1/m2 block has coefficient -9/4 before its common denominator, and the total expression adds the body-exchanged contribution; the earlier -3/2 coefficient is reported by the docstring.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SSS.py](../../../nrpypn/PN_Hamiltonian_SSS.py), `PN_Hamiltonian_SSS.f_H_SSS_3PN.f_H_SSS_3PN_pt`, final `H_SSS_3PN` assignment, and class docstring.
- Corroboration: [README.md](../../../README.md), `Validation`, identifies the cubic-spin coefficient and dependent orbital expressions as corrected. It is a separate repository account, not independent scientific authority for the coefficient.

The class docstring separately reports that Hergt and Schaefer,
arXiv:0809.2208, Eq. 2.13 prints prefactor `3*m2/(4*m1)` where
`3*m1/(4*m2)` is needed to reproduce Levi and Steinhoff, Eq. 3.12. This
reciprocal mass-prefactor warning is distinct from the `-3/2` to `-9/4`
grouping history above. Equal-mass comparisons cannot distinguish those
reciprocal ratios. The publications have not been independently checked here;
this page preserves the module's account rather than confirming a printed error.

Claim evidence:
- Claim: The cubic-spin class docstring reports a separate reciprocal mass-prefactor warning for Hergt and Schaefer Eq. 2.13, comparing printed 3*m2/(4*m1) with reported-needed 3*m1/(4*m2) to reproduce Levi and Steinhoff Eq. 3.12; the publications remain independently unverified here.
- Role: descriptive behavior
- Deciding authority: [PN_Hamiltonian_SSS.py](../../../nrpypn/PN_Hamiltonian_SSS.py), `PN_Hamiltonian_SSS` class docstring
- Corroboration: none available from independently verified publication evidence; the defining docstring is authority only for the warning it reports.

### Related orbital expressions

`PN_p_t.p_t__a_7`, `PN_MOmega.MOmega__a_7`, and `PN_p_r.f_p_r`
explicitly attribute their cubic-spin changes to this Hamiltonian coefficient.
Their closed expressions are written separately; changing the Hamiltonian
alone does not regenerate those formulas. In contrast, the radial module
constructs `Htot_xyplane_binary` by adding `H_SSS_3PN` and its other
Hamiltonian contributions directly.

Claim evidence:
- Claim: The separately written tangential, frequency, and closed radial formulas document dependence on the SSS coefficient, while the radial total Hamiltonian directly includes H_SSS_3PN.
- Role: descriptive behavior
- Deciding authority: [PN_p_t.py](../../../nrpypn/PN_p_t.py), `PN_p_t.p_t__a_7`; [PN_MOmega.py](../../../nrpypn/PN_MOmega.py), `PN_MOmega.MOmega__a_7`; [PN_p_r.py](../../../nrpypn/PN_p_r.py), `PN_p_r.f_p_r` and `f_Htot_xyplane_binary`.
- Corroboration: [README.md](../../../README.md), `Validation`, describes the same dependent expressions; it does not independently derive their coefficients.

The numerical `trusted_dict` contains a reference for `H_SSS_3PN`.
Regression agreement with that value checks the stored expression, not the
coefficient's scientific derivation.

## Sources

- [PN_Hamiltonian_SSS.py](../../../nrpypn/PN_Hamiltonian_SSS.py) — module and class docstrings, `PN_Hamiltonian_SSS.__init__`, and `f_H_SSS_3PN`.
- [PN_p_t.py](../../../nrpypn/PN_p_t.py) — `PN_p_t.p_t__a_7`.
- [PN_MOmega.py](../../../nrpypn/PN_MOmega.py) — `PN_MOmega.MOmega__a_7`.
- [PN_p_r.py](../../../nrpypn/PN_p_r.py) — `PN_p_r.f_Htot_xyplane_binary` and `f_p_r`.
- [README.md](../../../README.md) — `Validation`.
- [tests/PN_Hamiltonian_SSS.py](../../../nrpypn/tests/PN_Hamiltonian_SSS.py) — `trusted_dict`.

## See Also

- Parent: [Hamiltonians](index.md)
- See also: [Spin-spin](spin-spin.md)
- See also: [Tangential momentum and frequency](../orbital-quantities/tangential-momentum-and-frequency.md)
- See also: [Radial momentum](../orbital-quantities/radial-momentum.md)
- Contrasts with: [Scientific evidence](../../validation/scientific-evidence.md)
