"""
As documented in the NRPyPN notebook
PN-dE_GW_dt.ipynb, this Python script
generates dE_GW/dt at highest known
post-Newtonian order (as of 2015, at
least).

Author:  Zachariah B. Etienne
         zachetie **at** gmail **dot* com
"""
from typing import List, Tuple, cast
import sympy as sp  # SymPy: The Python computer algebra package upon which NRPy+ depends
import nrpy.indexedexp as ixp  # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import nrpy.validate_expressions.validate_expressions as ve

# NRPyPN: shortcuts for e.g., vector operations
from nrpypn.NRPyPN_shortcuts import div, dot, gamma_EulerMascheroni


class PN_dE_GW_dt_and_dM_dt:
    """
    This class calculates gravitational-wave energy flux and mass flux.
    Implements Eqs: A1-13 of https://arxiv.org/abs/1502.01747
    Eqs A22-28 of https://arxiv.org/abs/1502.01747, with
    Eq A.14 of https://arxiv.org/abs/0709.0093 for Mdot
           and correction on b[7] term by comparison with
     https://link.springer.com/content/pdf/10.12942/lrr-2014-2.pdf

    :param mOmega: Angular frequency
    :param m1: Mass of first object
    :param m2: Mass of second object
    :param n12U: List of symbolic expressions
    :param S1U: List of symbolic expressions for S1U
    :param S2U: List of symbolic expressions for S2U
    """

    def __init__(
        self,
        mOmega: sp.Expr,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
    ):
        self.dE_GW_dt_plus_dM_dt: sp.Expr = sp.sympify(0)
        self.dE_GW_dt: sp.Expr = sp.sympify(0)
        self.dM_dt: sp.Expr = sp.sympify(0)

        self.f_dE_GW_dt_and_dM_dt(mOmega, m1, m2, n12U, S1U, S2U)

    # Constants given in Eqs A1-13 of https://arxiv.org/abs/1502.01747
    def dE_GW_dt_OBKPSS2015_consts(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        _n12U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
    ) -> Tuple[
        sp.Expr, sp.Expr, List[sp.Expr], List[sp.Expr], List[sp.Expr], sp.Expr, sp.Expr
    ]:  # _n12U unused.
        """
        This method calculates and returns several constants based on given parameters.

        :param m1: Mass of first object
        :param m2: Mass of second object
        :param _n12U: Unused parameter for compatibility
        :param S1U: List of symbolic expressions for S1U
        :param S2U: List of symbolic expressions for S2U
        :return: Tuple of calculated constants and lists of symbolic expressions: nu, delta, l, chi_a, chi_s, s_l, sigma_l
        """
        # define scalars:
        m = m1 + m2
        nu = m1 * m2 / m**2
        delta = (m1 - m2) / m
        # define vectors:
        Stot = ixp.zerorank1()
        Sigma = ixp.zerorank1()
        l = ixp.zerorank1()
        l[2] = sp.sympify(1)
        chi1U = ixp.zerorank1()
        chi2U = ixp.zerorank1()
        chi_s = ixp.zerorank1()
        chi_a = ixp.zerorank1()
        for i in range(3):
            Stot[i] = S1U[i] + S2U[i]
            Sigma[i] = (m1 + m2) / m2 * S2U[i] - (m1 + m2) / m1 * S1U[i]
            chi1U[i] = S1U[i] / m1**2
            chi2U[i] = S2U[i] / m2**2
            chi_s[i] = div(1, 2) * (chi1U[i] + chi2U[i])
            chi_a[i] = div(1, 2) * (chi1U[i] - chi2U[i])
        # define scalars that depend on vectors
        s_l = dot(Stot, l) / m**2
        # s_n = dot(Stot,n12U)/m**2
        sigma_l = dot(Sigma, l) / m**2
        # sigma_n = dot(Sigma,n12U)/m**2
        return nu, delta, l, chi_a, chi_s, s_l, sigma_l

    def f_dE_GW_dt_and_dM_dt(
        self,
        mOmega: sp.Expr,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
    ) -> None:
        """
        This method calculates and updates the class attributes `dE_GW_dt_plus_dM_dt`, `dE_GW_dt`, `dM_dt`.
        Based on Eqs A22-28 of https://arxiv.org/abs/1502.01747, with
           Eq A.14 of https://arxiv.org/abs/0709.0093 for Mdot
           and correction on b[7] term by comparison with
        https://link.springer.com/content/pdf/10.12942/lrr-2014-2.pdf

        :param mOmega: Angular frequency
        :param m1: Mass of first object
        :param m2: Mass of second object
        :param n12U: List of symbolic expressions for n12U
        :param S1U: List of symbolic expressions for S1U
        :param S2U: List of symbolic expressions for S2U
        """

        def f_compute_quantities(
            mOmega: sp.Expr,
            m1: sp.Expr,
            m2: sp.Expr,
            n12U: List[sp.Expr],
            S1U: List[sp.Expr],
            S2U: List[sp.Expr],
            which_quantity: str,
        ) -> sp.Expr:
            if which_quantity not in ("dM_dt", "dE_GW_dt", "dE_GW_dt_plus_dM_dt"):
                raise ValueError(f"which_quantity == {which_quantity} not supported!")

            nu, delta, l, chi_a, chi_s, s_l, sigma_l = self.dE_GW_dt_OBKPSS2015_consts(
                m1, m2, n12U, S1U, S2U
            )
            x = (mOmega) ** div(2, 3)

            # fmt: off
            # Compute b_5_Mdot:
            b_5_Mdot = (-div(1,4)*(+(1-3*nu)*dot(chi_s,l)*(1+3*dot(chi_s,l)**2+9*dot(chi_a,l)**2)
                                +(1-  nu)*delta*dot(chi_a,l)*(1+3*dot(chi_a,l)**2+9*dot(chi_s,l)**2)))
            if which_quantity == "dM_dt":
                return cast(sp.Expr, div(32,5)*nu**2*x**5*b_5_Mdot*x**div(5,2))

            b = ixp.zerorank1(dimension=10)
            b[2] = -div(1247,336) - div(35,12)*nu
            b[3] = +4*sp.pi - 4*s_l - div(5,4)*delta*sigma_l
            b[4] =(-div(44711,9072) + div(9271,504)*nu + div(65,18)*nu**2
                   +(+div(287,96) + div( 1,24)*nu)*dot(chi_s,l)**2
                   -(+div( 89,96) + div( 7,24)*nu)*dot(chi_s,chi_s)
                   +(+div(287,96) -         12*nu)*dot(chi_a,l)**2
                   +(-div( 89,96) +          4*nu)*dot(chi_a,chi_a)
                   +div(287,48)*delta*dot(chi_s,l)*dot(chi_a,l) - div(89,48)*delta*dot(chi_s,chi_a))
            b[5] =(-div(8191,672)*sp.pi - div(9,2)*s_l - div(13,16)*delta*sigma_l
                   +nu*(-div(583,24)*sp.pi + div(272,9)*s_l + div(43,4)*delta*sigma_l))
            if which_quantity == "dE_GW_dt_plus_dM_dt":
                b[5]+= b_5_Mdot
            b[6] =(+div(6643739519,69854400) + div(16,3)*sp.pi**2 - div(1712,105)*gamma_EulerMascheroni
                   -div(856,105)*sp.log(16*x) + (-div(134543,7776) + div(41,48)*sp.pi**2)*nu
                   -div(94403,3024)*nu**2 - div(775,324)*nu**3 - 16*sp.pi*s_l - div(31,6)*sp.pi*delta*sigma_l)
            b[7] =(+(+div(476645,6804) + div(6172,189)*nu - div(2810,27)*nu**2)*s_l
                   +(+div(9535,336) + div(1849,126)*nu - div(1501,36)*nu**2)*delta*sigma_l
                   +(-div(16285,504) + div(214745,1728)*nu + div(193385,3024)*nu**2)*sp.pi)
            b[8] =(+(-div(3485,96)*sp.pi + div(13879,72)*sp.pi*nu)*s_l
                   +(-div(7163,672)*sp.pi + div(130583,2016)*sp.pi*nu)*delta*sigma_l)
            b_sum = sp.sympify(1)
            for k in range(9):
                b_sum += b[k]*x**div(k,2)
            return cast(sp.Expr, div(32,5)*nu**2*x**5*b_sum)
            # fmt: on

        self.dE_GW_dt_plus_dM_dt = f_compute_quantities(
            mOmega, m1, m2, n12U, S1U, S2U, "dE_GW_dt_plus_dM_dt"
        )
        self.dE_GW_dt = f_compute_quantities(mOmega, m1, m2, n12U, S1U, S2U, "dE_GW_dt")
        self.dM_dt = f_compute_quantities(mOmega, m1, m2, n12U, S1U, S2U, "dM_dt")


if __name__ == "__main__":
    import doctest
    import os
    import sys

    results = doctest.testmod()
    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")

    in_mOmega, in_m1, in_m2 = sp.symbols("mOmega m1 m2")
    in_n12U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("n12U"))
    in_S1U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("S1U"))
    in_S2U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("S2U"))
    mass_energy_fluxes = PN_dE_GW_dt_and_dM_dt(
        in_mOmega, in_m1, in_m2, in_n12U, in_S1U, in_S2U
    )
    results_dict = ve.process_dictionary_of_expressions(
        mass_energy_fluxes.__dict__, fixed_mpfs_for_free_symbols=True
    )
    ve.compare_or_generate_trusted_results(
        os.path.abspath(__file__),
        os.getcwd(),
        # File basename. If this is set to "trusted_module_test1", then
        #   trusted results_dict will be stored in tests/trusted_module_test1.py
        f"{os.path.splitext(os.path.basename(__file__))[0]}",
        results_dict,
    )
