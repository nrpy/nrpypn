"""
As documented in the NRPyPN notebook
PN-p_r.ipynb, this Python script
generates the expression for the radial
component of momentum p_r up to and
including terms at 3.5PN order.
It largely follows the technique of
 Ramos-Buades, Husa, and Pratten (2018)
   https://arxiv.org/abs/1810.00036

Author:  Zach Etienne
         zachetie **at** gmail **dot* com
"""

from typing import List, cast
import sympy as sp  # SymPy: The Python computer algebra package upon which NRPy+ depends
import nrpy.indexedexp as ixp
import nrpy.validate_expressions.validate_expressions as ve

# NRPyPN: shortcuts for e.g., vector operations
from nrpypn.NRPyPN_shortcuts import Pt, Pr, nU, div
from nrpypn.PN_p_t import PN_p_t
from nrpypn.PN_MOmega import PN_MOmega
from nrpypn.PN_dE_GW_dt_and_dM_dt import PN_dE_GW_dt_and_dM_dt
from nrpypn.PN_Hamiltonian_NS import PN_Hamiltonian_NS
from nrpypn.PN_Hamiltonian_SS import PN_Hamiltonian_SS
from nrpypn.PN_Hamiltonian_SSS import PN_Hamiltonian_SSS
from nrpypn.PN_Hamiltonian_SO import PN_Hamiltonian_SO


class PN_p_r:
    """

    Core class functions:
    f_dr_dt(Htot_xyplane_binary, m1,m2, n12U,n21U, chi1U,chi2U, S1U,S2U, p1U,p2U, r)
        Given Htot_xyplane_binary and other standard input parameters, compute
        dr_dt = dr/dt and store to class variable of the same name.

    f_p_r_fullHam(m1,m2, n12U,n21U, chi1U,chi2U, S1U,S2U, p1U,p2U, r)
          Compute p_r using the full Hamiltonian, without truncating
          higher-order terms self-consistently.

    f_p_r(m1,m2, chi1U,chi2U, r)
          Compute p_r and store to class variable of the same name.
    """

    def __init__(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        n21U: List[sp.Expr],
        chi1U: List[sp.Expr],
        chi2U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r: sp.Expr,
        use_alternative_approach_for_p_r: bool = False,
    ):
        self.Htot_xyplane_binary: sp.Expr = sp.sympify(0)
        self.dr_dt: sp.Expr = sp.sympify(0)
        self.p_r: sp.Expr = sp.sympify(0)

        if use_alternative_approach_for_p_r:
            self.f_p_r_alternative__use_fullHam(
                m1, m2, n12U, n21U, chi1U, chi2U, S1U, S2U, p1U, p2U, r
            )
        else:
            self.f_p_r(m1, m2, n12U, n21U, chi1U, chi2U, S1U, S2U, p1U, p2U, r)

    #################################
    #################################

    # Step 1: Construct full Hamiltonian
    #         expression for a binary instantaneously
    #         orbiting on the xy plane, store
    #         result to Htot_xyplane_binary
    def f_Htot_xyplane_binary(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        n21U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r: sp.Expr,
    ) -> None:
        """
        Given standard input parameters, compute
        the Hamiltonian for a binary system
        orbiting instantaneously on the xy plane,
        and store to the class variable
        Htot_xyplane_binary
        """

        def make_replacements(expr: sp.Expr) -> sp.Expr:
            zero = sp.sympify(0)
            one = sp.sympify(1)
            return cast(
                sp.Expr,
                expr.subs(p1U[1], Pt)
                .subs(p2U[1], -Pt)
                .subs(p1U[2], zero)
                .subs(p2U[2], zero)
                .subs(p1U[0], -Pr)
                .subs(p2U[0], Pr)
                .subs(nU[0], one)
                .subs(nU[1], zero)
                .subs(nU[2], zero),
            )

        H_NS = PN_Hamiltonian_NS(m1, m2, p1U, n12U, r)

        self.Htot_xyplane_binary = make_replacements(
            H_NS.H_Newt + H_NS.H_NS_1PN + H_NS.H_NS_2PN + H_NS.H_NS_3PN
        )

        H_SO = PN_Hamiltonian_SO(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r)
        self.Htot_xyplane_binary += make_replacements(
            H_SO.H_SO_1p5PN + H_SO.H_SO_2p5PN + H_SO.H_SO_3p5PN
        )

        H_SS = PN_Hamiltonian_SS(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r)
        self.Htot_xyplane_binary += make_replacements(
            H_SS.H_SS_2PN + H_SS.H_SS_S1S2_3PN + H_SS.H_SS_S1sq_S2sq_3PN
        )

        H_SSS = PN_Hamiltonian_SSS(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r)
        self.Htot_xyplane_binary += make_replacements(H_SSS.H_SSS_3PN)

    # Function for computing dr/dt
    def f_dr_dt(
        self,
        Htot_xyplane_binary: sp.Expr,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        chi1U: List[sp.Expr],
        chi2U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        r: sp.Expr,
    ) -> None:
        """
        Given Htot_xyplane_binary (computed
        above) and other standard input
        parameters, compute
        dr_dt = dr/dt and store to class
        variable of the same name.
        """
        # First compute p_t
        pt = PN_p_t(m1, m2, chi1U, chi2U, r)

        # Then compute dH_{circ}/dr = partial_H(p_r=0)/partial_r
        #                                  + partial_H(p_r=0)/partial_{p_t} partial_{p_t}/partial_r
        dHcirc_dr = +sp.diff(Htot_xyplane_binary.subs(Pr, sp.sympify(0)), r) + sp.diff(
            Htot_xyplane_binary.subs(Pr, sp.sympify(0)), Pt
        ) * sp.diff(pt.p_t, r)

        # Then compute M Omega
        MOm = PN_MOmega(m1, m2, chi1U, chi2U, r)

        # Next compute dE_GW_dt_plus_dM_dt
        dE_GW_dt_and_dM_dt = PN_dE_GW_dt_and_dM_dt(MOm.MOmega, m1, m2, n12U, S1U, S2U)
        dE_GW_dt_plus_dM_dt = dE_GW_dt_and_dM_dt.dE_GW_dt_plus_dM_dt

        # Finally, compute dr/dt
        self.dr_dt = dE_GW_dt_plus_dM_dt / dHcirc_dr

    # Next we compute p_r as a function of dr_dt (unknown) and known quantities using
    # p_r  approx [dr/dt - (partial_H/partial_{p_r})|_{p_r=0}] * [(partial^2_{H}/partial_{p_r^2})|_{p_r=0}]^{-1}
    def f_p_r_alternative__use_fullHam(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        n21U: List[sp.Expr],
        chi1U: List[sp.Expr],
        chi2U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r: sp.Expr,
    ) -> None:
        """
        Basic equation for p_r can be written in the
          form:
         p_r  approx [dr/dt - (partial_H/partial_{p_r})|_{p_r=0}] * [(partial^2_{H}/partial_{p_r^2})|_{p_r=0}]^{-1},
          where
        dr/dt = [dE_{rm GW}/dt + dM/dt] * [dH_{circ} / dr]^{-1},
         and
        H_{circ} = Htot_xyplane_binary|_{p_r=0}
         -> [dH_{circ}(r,p_t(r)) / dr] = partial_{H(p_r=0)}/partial_r
                   + partial_{H(p_r=0)}/partial_{p_t} partial_{p_t}/partial_r.
         Here,
         * the expression for p_t is given by PN_p_t.py
         * the expression for [dE_{rm GW}/dt + dM/dt] is given by
                   PN_dE_GW_dt_and_dM_dt.py
           + Since [dE_{rm GW}/dt + dM/dt] is a function of MOmega,
           we also need input from the PN_MOmega.py Python module.
        """
        self.f_Htot_xyplane_binary(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r)
        self.f_dr_dt(self.Htot_xyplane_binary, m1, m2, n12U, chi1U, chi2U, S1U, S2U, r)

        dHdpr_przero = sp.diff(self.Htot_xyplane_binary, Pr).subs(Pr, sp.sympify(0))
        d2Hdpr2_przero = sp.diff(sp.diff(self.Htot_xyplane_binary, Pr), Pr).subs(
            Pr, sp.sympify(0)
        )
        self.p_r = (self.dr_dt - dHdpr_przero) / (d2Hdpr2_przero)

    def f_p_r(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        n21U: List[sp.Expr],
        chi1U: List[sp.Expr],
        chi2U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r: sp.Expr,
    ) -> None:
        """
        Ramos-Buades, Husa, and Pratten (2018)
          approach for computing p_r.
        Transcribed from Eq 2.18 of
        Ramos-Buades, Husa, and Pratten (2018),
          https://arxiv.org/abs/1810.00036
        """
        q = m2 / m1  # It is assumed that q >= 1, so m2 >= m1.
        self.f_Htot_xyplane_binary(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r)
        self.f_dr_dt(self.Htot_xyplane_binary, m1, m2, n12U, chi1U, chi2U, S1U, S2U, r)
        chi1x = chi1U[0]
        chi1y = chi1U[1]
        chi1z = chi1U[2]
        chi2x = chi2U[0]
        chi2y = chi2U[1]
        chi2z = chi2U[2]
        # fmt: off
        p_r_num = (-self.dr_dt
                   +(-(6*q+13)*q**2*chi1x*chi2y/(4*(q+1)**4)
                     -(6*q+ 1)*q**2*chi2x*chi2y/(4*(q+1)**4)
                     +chi1y*(-q*(   q+6)*chi1x/(4*(q+1)**4)
                             -q*(13*q+6)*chi2x/(4*(q+1)**4)))/r**div(7,2)
                   +(+chi1z*(+3*q   *(5*q+2)*chi1x*chi2y/(2*(q+1)**4)
                             -3*q**2*(2*q+5)*chi2x*chi2y/(2*(q+1)**4))
                     +chi1y*chi2z*(+3*q**2*(2*q+5)*chi2x/(2*(q+1)**4)
                                   -3*q   *(5*q+2)*chi1x/(2*(q+1)**4)))/r**4)
        p_r_den = (-(q+1)**2/q - (-7*q**2-15*q-7)/(2*q*r)
                   -(47*q**4 + 229*q**3 + 363*q**2 + 229*q + 47)/(8*q*(q+1)**2*r**2)
                   -(+( 4*q**2 + 11*q + 12)*chi1z/(4*q*(q+1))
                     +(12*q**2 + 11*q +  4)*chi2z/(4*  (q+1)))/r**div(5,2)
                   -(+(- 53*q**5 - 357*q**4 - 1097*q**3 - 1486*q**2 - 842*q - 144)*chi1z/(16*q*(q+1)**4)
                     +(-144*q**5 - 842*q**4 - 1486*q**3 - 1097*q**2 - 357*q -  53)*chi2z/(16  *(q+1)**4))/r**div(7,2)
                   -(+(  q**2 + 9*q + 9)*chi1x**2/(2*q*(q+1)**2)
                     +(3*q**2 + 5*q + 3)*chi2x*chi1x/((q+1)**2)
                     +(3*q**2 + 8*q + 3)*chi1y*chi2y/(2*(q+1)**2)
                     -9*q**2*chi2y**2/(4*(q+1))
                     +(3*q**2 + 8*q + 3)*chi1z*chi2z/(2*(q+1)**2)
                     -9*q**2*chi2z**2/(4*(q+1))
                     +(9*q**3 + 9*q**2 + q)*chi2x**2/(2*(q+1)**2)
                     +(-363*q**6 - 2608*q**5 - 7324*q**4 - 10161*q**3 - 7324*q**2 - 2608*q - 363)/(48*q*(q+1)**4)
                     -9*chi1y**2/(4*q*(q+1))
                     -9*chi1z**2/(4*q*(q+1)) - sp.pi**2/16)/r**3)
        # fmt: on
        self.p_r = p_r_num / p_r_den


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

    in_m1, in_m2, in_r12 = sp.symbols("m1 m2 r12")
    in_n12U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("n12U"))
    in_n21U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("n21U"))
    in_chi1U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("chi1U"))
    in_chi2U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("chi2U"))
    in_S1U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("S1U"))
    in_S2U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("S2U"))
    in_p1U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("p1U"))
    in_p2U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("p2U"))
    in_nU: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("nU"))
    p_r = PN_p_r(
        in_m1,
        in_m2,
        in_n12U,
        in_n21U,
        in_chi1U,
        in_chi2U,
        in_S1U,
        in_S2U,
        in_p1U,
        in_p2U,
        in_r12,
    )
    results_dict = ve.process_dictionary_of_expressions(
        p_r.__dict__, fixed_mpfs_for_free_symbols=True
    )
    ve.compare_or_generate_trusted_results(
        os.path.abspath(__file__),
        os.getcwd(),
        # File basename. If this is set to "trusted_module_test1", then
        #   trusted results_dict will be stored in tests/trusted_module_test1.py
        f"{os.path.splitext(os.path.basename(__file__))[0]}",
        results_dict,
    )
