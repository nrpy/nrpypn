"""
As documented in the NRPyPN notebook
PN-Hamiltonian-SSS.ipynb, this Python script
generates spin-spin-spin coupling pieces of the
post-Newtonian (PN) Hamiltonian, up to and
including 3PN order.

Author:  Zachariah B. Etienne
         zachetie **at** gmail **dot* com
"""
from typing import List, cast
import sympy as sp
import nrpy.indexedexp as ixp  # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import nrpy.validate_expressions.validate_expressions as ve

# NRPyPN: shortcuts for e.g., vector operations
from nrpypn.NRPyPN_shortcuts import div, dot, cross


#################################
class PN_Hamiltonian_SSS:
    """
    Implement 3PN spin-spin-spin term, from Eq. 3.12 of
            Levi and Steinhoff (2015):
         https://arxiv.org/abs/1410.2601

    :param m1: Mass of the first object
    :param m2: Mass of the second object
    :param n12U: Unit vector from the first to the second object
    :param n21U: Unit vector from the second to the first object
    :param S1U: Spin vector of the first object
    :param S2U: Spin vector of the second object
    :param p1U: Momentum of the first object
    :param p2U: Momentum of the second object
    :param r12: Separation between the two objects
    """

    def __init__(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        n21U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> None:
        """Initialize class by calling f_H_SSS_3PN."""
        self.f_H_SSS_3PN(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r12)

    def f_H_SSS_3PN(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        n21U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> None:
        """3PN spin-spin-spin term, from Eq. 3.12 of Levi and Steinhoff (2015):
        https://arxiv.org/abs/1410.2601"""

        def f_H_SSS_3PN_pt(
            m1: sp.Expr,
            m2: sp.Expr,
            nU: List[sp.Expr],
            S1U: List[sp.Expr],
            S2U: List[sp.Expr],
            p1U: List[sp.Expr],
            p2U: List[sp.Expr],
            r: sp.Expr,
        ) -> sp.Expr:
            """Implement one part of the spin-spin-spin, call once for particle 1 acting on 2,
            and again for vice-versa"""
            p2_minus_m2_over_4m1_p1 = ixp.zerorank1()
            for i in range(3):
                p2_minus_m2_over_4m1_p1[i] = p2U[i] - m2 / (4 * m1) * p1U[i]
            # fmt: off
            H_SSS_3PN_pt = (div(3,2)*(dot(S1U,S1U)*dot(S2U,cross(nU,p1U))
                                       +dot(S1U,nU)*dot(S2U,cross(S1U,p1U))
                                       -5*dot(S1U,nU)**2*dot(S2U,cross(nU,p1U))
                                       +dot(nU,cross(S1U,S2U))*( dot(S1U,p1U)
                                                                -5*dot(S1U,nU)*dot(p1U,nU)))
                            -3*m1/(2*m2)*(   dot(S1U,S1U)  *dot(S2U,cross(nU,p2U))
                                          +2*dot(S1U,nU)   *dot(S2U,cross(S1U,p2U))
                                          -5*dot(S1U,nU)**2*dot(S2U,cross(nU,p2U)))
                            -dot(cross(S1U,nU),p2_minus_m2_over_4m1_p1)*(dot(S1U,S1U) - 5*dot(S1U,nU)**2))/(m1**2*r**4)
            # fmt: on
            return cast(sp.Expr, H_SSS_3PN_pt)

        self.H_SSS_3PN = +f_H_SSS_3PN_pt(
            m1, m2, n12U, S1U, S2U, p1U, p2U, r12
        ) + f_H_SSS_3PN_pt(m2, m1, n21U, S2U, S1U, p2U, p1U, r12)


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
    in_S1U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("S1U"))
    in_S2U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("S2U"))
    in_p1U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("p1U"))
    in_p2U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("p2U"))
    spin_spin_spin_Hamiltonian = PN_Hamiltonian_SSS(
        in_m1, in_m2, in_n12U, in_n21U, in_S1U, in_S2U, in_p1U, in_p2U, in_r12
    )
    results_dict = ve.process_dictionary_of_expressions(
        spin_spin_spin_Hamiltonian.__dict__, fixed_mpfs_for_free_symbols=True
    )
    ve.compare_or_generate_trusted_results(
        os.path.abspath(__file__),
        os.getcwd(),
        # File basename. If this is set to "trusted_module_test1", then
        #   trusted results_dict will be stored in tests/trusted_module_test1.py
        f"{os.path.splitext(os.path.basename(__file__))[0]}",
        results_dict,
    )
