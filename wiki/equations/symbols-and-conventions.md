# Symbols and Conventions

> Resolve notation before substituting into the symbolic expressions. · Status: confirmed
> Up: [Equations](index.md)

## Summary

Read `q` in its defining module: it can denote separation or the mass ratio.
The numerical evaluator normalizes total mass and converts dimensionless spins
to spin angular momenta. Hamiltonian classes can accept independent body
momenta; the quasicircular evaluation path supplies additional frame and sign
choices.

## Detail

### Separation and mass ratio

The global `r` and `q` in `NRPyPN_shortcuts` are separation symbols: `num_eval`
substitutes `nr` for both. By contrast, local `q = m2 / m1` in `PN_p_t`,
`PN_MOmega`, and the closed radial expression denotes mass ratio. In this KB,
\(Q=m_2/m_1\) denotes that ratio when separation and mass ratio occur together.
The numerical input is named `qmassratio`; its documented domain is
`qmassratio >= 1`, but `num_eval` does not check that inequality.

Claim evidence:
- Claim: Global shortcut q denotes separation, while the orbital-series local q denotes m2/m1; num_eval substitutes nr for the separation and documents, without enforcing, qmassratio >= 1.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), global `r, q` declaration and `num_eval`; [PN_p_t.py](../../nrpypn/PN_p_t.py), `PN_p_t.f_p_t`; [PN_MOmega.py](../../nrpypn/PN_MOmega.py), `PN_MOmega.MOmega__a_2_thru_a_4`; [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_p_r`.
- Corroboration: [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py), `eval__P_t__and__P_r`, passes `qmassratio` and `nr` separately to `num_eval`; it provides no separate domain check.

### Mass and spin normalization

For numerical evaluation, the substitutions are

\[
m_1=\frac{1}{1+Q},\qquad m_2=\frac{Q}{1+Q},\qquad
m_1+m_2=1,\qquad \mathbf S_A=m_A^2\boldsymbol\chi_A.
\]

`S1U` and `S2U` represent spin angular momenta in the Hamiltonians, whereas
`chi1U` and `chi2U` supply dimensionless spin components to the orbital series.
These substitutions describe `num_eval`; they are not constraints imposed on
every symbolic class constructor.

Claim evidence:
- Claim: num_eval sets total mass to unity and substitutes S_A = m_A^2 chi_A; the symbolic Hamiltonian constructors accept masses and spin vectors as supplied.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `num_eval`, assignments to `nm1`, `nm2`, and `nS1U0` through `nS2U2`; [PN_Hamiltonian_SO.py](../../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO.__init__`.
- Corroboration: [PN_dE_GW_dt_and_dM_dt.py](../../nrpypn/PN_dE_GW_dt_and_dM_dt.py), `PN_dE_GW_dt_and_dM_dt.dE_GW_dt_OBKPSS2015_consts`, reconstructs `chi1U = S1U / m1**2` and its body-2 counterpart.

### Direction vectors and momentum restrictions

The spin Hamiltonian docstrings define
\(\mathbf n_{12}=(\mathbf x_1-\mathbf x_2)/r_{12}\) and
\(\mathbf n_{21}=-\mathbf n_{12}\). The shortcuts set
`n12U = nU`, `n21U = -nU`, `p1U = pU`, and `p2U = -pU` componentwise.
The spin Hamiltonian constructors themselves receive `p1U` and `p2U`
independently; supplying the shortcuts is what restricts those expressions to
opposite momenta.

Claim evidence:
- Claim: The shortcut vectors impose opposite momenta and opposite separation directions; the spin Hamiltonian constructors do not independently enforce these relations.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), derived `n12U`, `n21U`, `p1U`, and `p2U`; [PN_Hamiltonian_SO.py](../../nrpypn/PN_Hamiltonian_SO.py), `PN_Hamiltonian_SO` parameter docstring and `__init__`.
- Corroboration: [PN_Hamiltonian_SS.py](../../nrpypn/PN_Hamiltonian_SS.py), `PN_Hamiltonian_SS.__init__`, and [PN_Hamiltonian_SSS.py](../../nrpypn/PN_Hamiltonian_SSS.py), `PN_Hamiltonian_SSS.__init__`, also accept the body vectors separately.

With those shortcuts, `PN_p_r.f_Htot_xyplane_binary` makes the specialization
\(\mathbf n=(1,0,0)\),
\(\mathbf p_1=(-P_r,P_t,0)\), and
\(\mathbf p_2=(P_r,-P_t,0)\). Thus positive `Pr` points inward for body 1
on the positive x-axis. Its attribute `dr_dt` denotes \(-dr/dt\), not the
outward separation velocity. The replacement function targets the global
`nU`; independently named input directions do not automatically receive this
orientation.

Claim evidence:
- Claim: With the shortcut vector relations, the radial module uses inward-positive Pr and dr_dt = -dr/dt in an instantaneous xy-plane frame; its orientation substitutions target the global nU symbols.
- Role: descriptive behavior
- Deciding authority: [PN_p_r.py](../../nrpypn/PN_p_r.py), `PN_p_r.f_Htot_xyplane_binary.make_replacements` and `PN_p_r.f_dr_dt`.
- Corroboration: [quasicirc.py](../../nrpypn/quasicirc.py), `main`, places body 1 at positive x and assigns its radial momentum component `-|P_r|` in the printed initial-data instructions.

### Vector operations

`dot` sums component products, `cross` uses the three-dimensional Levi-Civita
symbol, and `div(a, b)` constructs an exact SymPy rational. These helpers do
not normalize vectors or insert a curved-space metric.

Claim evidence:
- Claim: dot and cross perform Cartesian component operations without vector normalization, and div constructs sp.Rational.
- Role: descriptive behavior
- Deciding authority: [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py), `dot`, `cross`, and `div`.
- Corroboration: none available; the defining helper expressions directly specify these operations, and no separate helper test is registered.

## Sources

- [NRPyPN_shortcuts.py](../../nrpypn/NRPyPN_shortcuts.py) — global symbol declarations, derived vectors, `dot`, `cross`, `div`, and `num_eval`.
- [PN_p_t.py](../../nrpypn/PN_p_t.py) — `PN_p_t.f_p_t`.
- [PN_MOmega.py](../../nrpypn/PN_MOmega.py) — `PN_MOmega.MOmega__a_2_thru_a_4`.
- [PN_p_r.py](../../nrpypn/PN_p_r.py) — `PN_p_r.f_Htot_xyplane_binary`, `f_dr_dt`, and `f_p_r`.
- [PN_Hamiltonian_SO.py](../../nrpypn/PN_Hamiltonian_SO.py) — `PN_Hamiltonian_SO` parameter docstring and `__init__`.
- [PN_Hamiltonian_SS.py](../../nrpypn/PN_Hamiltonian_SS.py) — `PN_Hamiltonian_SS.__init__`.
- [PN_Hamiltonian_SSS.py](../../nrpypn/PN_Hamiltonian_SSS.py) — `PN_Hamiltonian_SSS.__init__`.
- [PN_dE_GW_dt_and_dM_dt.py](../../nrpypn/PN_dE_GW_dt_and_dM_dt.py) — `PN_dE_GW_dt_and_dM_dt.dE_GW_dt_OBKPSS2015_consts`.
- [eval_p_t_and_p_r.py](../../nrpypn/eval_p_t_and_p_r.py) — `eval__P_t__and__P_r`.
- [quasicirc.py](../../nrpypn/quasicirc.py) — `main`, frame and momentum instructions.

## See Also

- Parent: [Equations](index.md)
- Depends on: [Numerical evaluation](../usage/numerical-evaluation.md)
- See also: [Radial momentum](orbital-quantities/radial-momentum.md)
- Contrasts with: [Scientific evidence](../validation/scientific-evidence.md)
