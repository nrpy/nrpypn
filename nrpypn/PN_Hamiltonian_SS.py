"""
As documented in the NRPyPN notebook
PN-Hamiltonian-Spin-Spin.ipynb, this Python script
generates spin-spin coupling pieces of the
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


class PN_Hamiltonian_SS:
    """
    This Python class generates spin-spin coupling pieces of the
    post-Newtonian (PN) Hamiltonian, up to and including 3PN order.

    :param m1: Mass of object 1.
    :param m2: Mass of object 2.
    :param n12U: Unit vector from object 1 to object 2.
    :param n21U: Unit vector from object 2 to object 1.
    :param S1U: Spin vector of object 1.
    :param S2U: Spin vector of object 2.
    :param p1U: Momentum vector of object 1.
    :param p2U: Momentum vector of object 2.
    :param r12: Distance between object 1 and object 2.
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
        # f_H_SS_2PN(m1,m2, S1U,S2U, nU, r12):
        #       Compute the complete H_SS_2PN term and store to
        #                     global variable of the same name.
        # nU: as defined in NRPyPN_shortcuts:
        #  n12U[i] = +nU[i]
        self.f_H_SS_2PN(m1, m2, S1U, S2U, n12U, r12)
        # f_H_SS_S1S2_3PN(m1,m2, n12U, S1U,S2U, p1U,p2U, r12):
        #       Compute HS1S2_3PN and store to global variable
        #                     of the same name.
        self.f_H_SS_S1S2_3PN(m1, m2, n12U, S1U, S2U, p1U, p2U, r12)
        # f_H_SS_S1sq_S2sq_3PN(m1,m2, n12U,n21U, S1U,S2U, p1U,p2U, r12):
        #       Compute H_SS_S1sq_S2sq_3PN and store to global
        #                     variable of the same name.
        self.f_H_SS_S1sq_S2sq_3PN(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r12)

    def f_H_SS_2PN(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        nU: List[sp.Expr],
        q: sp.Expr,
    ) -> None:
        """
        2PN spin-spin term, from Eqs. 2.18 and 2.19 of
             Buonanno, Chen, and Damour (2006):
           https://arxiv.org/abs/gr-qc/0508067
        """
        S0U = ixp.zerorank1()
        for i in range(3):
            S0U[i] = (1 + m2 / m1) * S1U[i] + (1 + m1 / m2) * S2U[i]
        mu = m1 * m2 / (m1 + m2)
        self.H_SS_2PN = (
            mu / (m1 + m2) * (3 * dot(S0U, nU) ** 2 - dot(S0U, S0U)) / (2 * q**3)
        )

    #################################
    #################################
    def f_H_SS_S1S2_3PN(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        S1U: List[sp.Expr],
        S2U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> None:
        """
        3PN spin-spin S_1,S_2 coupling term, from Eq. 2.11 of
              Steinhoff, Hergt, and Sch\"afer (2008a)
                https://arxiv.org/abs/0712.1716
        """
        # fmt: off
        H_SS_S1S2_3PN = (+div(3,2)*(dot(cross(p1U,S1U),n12U)*dot(cross(p2U,S2U),n12U))
                         +       6*(dot(cross(p2U,S1U),n12U)*dot(cross(p1U,S2U),n12U))
                         -15*dot(S1U,n12U)*dot(S2U,n12U)*dot(p1U,n12U)*dot(p2U,n12U)
                         -3*dot(S1U,n12U)*dot(S2U,n12U)*dot(p1U,p2U)
                         +3*dot(S1U,p2U)*dot(S2U,n12U)*dot(p1U,n12U)
                         +3*dot(S2U,p1U)*dot(S1U,n12U)*dot(p2U,n12U)
                         +3*dot(S1U,p1U)*dot(S2U,n12U)*dot(p2U,n12U)
                         +3*dot(S2U,p2U)*dot(S1U,n12U)*dot(p1U,n12U)
                         -div(1,2)*dot(S1U,p2U)*dot(S2U,p1U)
                         +dot(S1U,p1U)*dot(S2U,p2U)
                         -3*dot(S1U,S2U)*dot(p1U,n12U)*dot(p2U,n12U)
                         +div(1,2)*dot(S1U,S2U)*dot(p1U,p2U))/(2*m1*m2*r12**3)
        H_SS_S1S2_3PN +=(-dot(cross(p1U,S1U),n12U)*dot(cross(p1U,S2U),n12U)
                         +dot(S1U,S2U)*dot(p1U,n12U)**2
                         -dot(S1U,n12U)*dot(S2U,p1U)*dot(p1U,n12U))*3/(2*m1**2*r12**3)
        H_SS_S1S2_3PN +=(-dot(cross(p2U,S2U),n12U)*dot(cross(p2U,S1U),n12U)
                         +dot(S1U,S2U)*dot(p2U,n12U)**2
                         -dot(S2U,n12U)*dot(S1U,p1U)*dot(p2U,n12U))*3/(2*m2**2*r12**3)
        H_SS_S1S2_3PN +=( dot(S1U,S2U)-2*dot(S1U,n12U)*dot(S2U,n12U))*6*(m1+m2)/r12**4
        self.H_SS_S1S2_3PN = H_SS_S1S2_3PN
        # fmt: on

    #################################
    #################################
    def f_H_SS_S1sq_S2sq_3PN(
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
        """
        # 3PN spin-orbit coupling term, from Eq. 9 of
        #    Steinhoff, Hergt, and Sch\"afer (2008b)
        #       https://arxiv.org/abs/0809.2200
        """

        def f_H_SS_particle(
            m1: sp.Expr,
            m2: sp.Expr,
            n12U: List[sp.Expr],
            S1U: List[sp.Expr],
            _S2U: List[sp.Expr],  # _S2U unused.
            p1U: List[sp.Expr],
            p2U: List[sp.Expr],
            r12: sp.Expr,
        ) -> sp.Expr:
            """Per-object H_SS contribution, call once to compute 1 on 2 and again for 2 on 1"""
            # fmt: off
            H_SS_S1sq_S2sq_3PN_particle = (
                +  m2/(4*m1**3)*dot(p1U,S1U)**2
                +3*m2/(8*m1**3)*dot(p1U,n12U)**2*dot(S1U,S1U)
                -3*m2/(8*m1**3)*dot(p1U,p1U)*dot(S1U,n12U)**2
                -3*m2/(4*m1**3)*dot(p1U,n12U)*dot(S1U,n12U)*dot(p1U,S1U)
                -3/(4*m1*m2)*dot(p2U,p2U)*dot(S1U,S1U)
                +9/(4*m1*m2)*dot(p2U,p2U)*dot(S1U,n12U)**2
                +3/(4*m1**2)*dot(p1U,p2U)*dot(S1U,S1U)
                -9/(4*m1**2)*dot(p1U,p2U)*dot(S1U,n12U)**2
                -3/(2*m1**2)*dot(p1U,n12U)*dot(p2U,S1U)*dot(S1U,n12U)
                +3/(m1**2)  *dot(p2U,n12U)*dot(p1U,S1U)*dot(S1U,n12U)
                +3/(4*m1**2)*dot(p1U,n12U)*dot(p2U,n12U)*dot(S1U,S1U)
                -15/(4*m1**2)*dot(p1U,n12U)*dot(p2U,n12U)*dot(S1U,n12U)**2)/r12**3
            H_SS_S1sq_S2sq_3PN_particle+= -(+div(9,2)*dot(S1U,n12U)**2
                                            -div(5,2)*dot(S1U,S1U)
                                            +7*m2/m1*dot(S1U,n12U)**2
                                            -3*m2/m1*dot(S1U,S1U))*m2/r12**4
            # fmt: on
            return cast(sp.Expr, H_SS_S1sq_S2sq_3PN_particle)

        # fmt: off
        self.H_SS_S1sq_S2sq_3PN = (+f_H_SS_particle(m1,m2, n12U, S1U,S2U, p1U,p2U, r12)
                                   +f_H_SS_particle(m2,m1, n21U, S2U,S1U, p2U,p1U, r12))
        # fmt: on


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
    spin_spin_Hamiltonian = PN_Hamiltonian_SS(
        in_m1, in_m2, in_n12U, in_n21U, in_S1U, in_S2U, in_p1U, in_p2U, in_r12
    )
    results_dict = ve.process_dictionary_of_expressions(
        spin_spin_Hamiltonian.__dict__, fixed_mpfs_for_free_symbols=True
    )
    ve.compare_or_generate_trusted_results(
        os.path.abspath(__file__),
        os.getcwd(),
        # File basename. If this is set to "trusted_module_test1", then
        #   trusted results_dict will be stored in tests/trusted_module_test1.py
        f"{os.path.splitext(os.path.basename(__file__))[0]}",
        results_dict,
    )
