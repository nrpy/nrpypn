"""
As documented in the NRPyPN notebook
PN-Hamiltonian-Spin-Orbit.ipynb, this Python script
generates spin-orbit coupling pieces of the
post-Newtonian (PN) Hamiltonian, up to and
including 3.5PN order.

Author:  Zachariah B. Etienne
         zachetie **at** gmail **dot* com
"""
from typing import List, cast
import sympy as sp
import nrpy.indexedexp as ixp  # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import nrpy.validate_expressions.validate_expressions as ve

# NRPyPN: shortcuts for e.g., vector operations
from nrpypn.NRPyPN_shortcuts import div, dot, cross


class PN_Hamiltonian_SO:
    """
    Class to compute post-Newtonian Hamiltonians for Spin-Orbit coupling terms.

    The Hamiltonians are calculated up to 3.5PN order based on Damour, Jaranowski, and Schäfer (2008).
    Reference: https://arxiv.org/abs/0711.1048

    Attributes:
    H_SO_1p5PN : sympy.Expr
        1.5PN Hamiltonian term
    H_SO_2p5PN : sympy.Expr
        2.5PN Hamiltonian term
    H_SO_3p5PN : sympy.Expr
        3.5PN Hamiltonian term

    :param m1: Mass of the first object
    :param m2: Mass of the second object
    :param n12U: Unit vector from object 1 to object 2
    :param n21U: Unit vector from object 2 to object 1
    :param S1U: Spin vector of object 1
    :param S2U: Spin vector of object 2
    :param p1U: Momentum vector of object 1
    :param p2U: Momentum vector of object 2
    :param r12: Scalar distance between object 1 and object 2
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
    ):
        self.H_SO_1p5PN: sp.Expr = sp.sympify(0)
        self.H_SO_2p5PN: sp.Expr = sp.sympify(0)
        self.H_SO_3p5PN: sp.Expr = sp.sympify(0)

        self.f_H_SO_1p5PN(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r12)
        self.f_H_SO_2p5PN(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r12)
        self.f_H_SO_3p5PN(m1, m2, n12U, n21U, S1U, S2U, p1U, p2U, r12)

    def f_H_SO_1p5PN(
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
        1.5PN spin-orbit coupling term, from Eq. 4.11a of
           Damour, Jaranowski, and Schäfer (2008)
             https://arxiv.org/abs/0711.1048
        """

        def f_Omega1(
            m1: sp.Expr,
            m2: sp.Expr,
            n12U: List[sp.Expr],
            p1U: List[sp.Expr],
            p2U: List[sp.Expr],
            r12: sp.Expr,
        ) -> List[sp.Expr]:
            """Compute Omega vector"""
            Omega1 = ixp.zerorank1()
            for i in range(3):
                Omega1[i] = (
                    div(3, 2) * m2 / m1 * cross(n12U, p1U)[i] - 2 * cross(n12U, p2U)[i]
                ) / r12**2
            return Omega1

        Omega1 = f_Omega1(m1, m2, n12U, p1U, p2U, r12)
        Omega2 = f_Omega1(m2, m1, n21U, p2U, p1U, r12)
        self.H_SO_1p5PN = dot(Omega1, S1U) + dot(Omega2, S2U)

    #################################
    #################################

    def f_H_SO_2p5PN(
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
        2.5PN spin-orbit coupling term, from Eq. 4.11b of
           Damour, Jaranowski, and Schäfer (2008)
             https://arxiv.org/abs/0711.1048
        """

        def f_Omega_SO_2p5PN(
            m1: sp.Expr,
            m2: sp.Expr,
            n12U: List[sp.Expr],
            p1U: List[sp.Expr],
            p2U: List[sp.Expr],
            r12: sp.Expr,
        ) -> List[sp.Expr]:
            """Compute Omega vector"""
            Omega1 = ixp.zerorank1()
            # fmt: off
            for i in range(3):
                Omega1[i] = (+(+(-div(11, 2) * m2 - 5 * m2 ** 2 / m1) * cross(n12U, p1U)[i]
                               + (6 * m1 + div(15, 2) * m2) * cross(n12U, p2U)[i]) / r12 ** 3
                             + (+(-div(5, 8) * m2 * dot(p1U, p1U) / m1 ** 3
                                  - div(3, 4) * dot(p1U, p2U) / m1 ** 2
                                  + div(3, 4) * dot(p2U, p2U) / (m1 * m2)
                                  - div(3, 4) * dot(n12U, p1U) * dot(n12U, p2U) / m1 ** 2
                                  - div(3, 2) * dot(n12U, p2U) ** 2 / (m1 * m2)) * cross(n12U, p1U)[i]
                                + (dot(p1U, p2U) / (m1 * m2) + 3 * dot(n12U, p1U) * dot(n12U, p2U) / (m1 * m2)) *
                                cross(n12U, p2U)[i]
                                + (div(3, 4) * dot(n12U, p1U) / m1 ** 2 - 2 * dot(n12U, p2U) / (m1 * m2)) * cross(p1U, p2U)[
                                    i]) / r12 ** 2)
            # fmt: on
            return Omega1

        Omega1_2p5PNU = f_Omega_SO_2p5PN(m1, m2, n12U, p1U, p2U, r12)
        Omega2_2p5PNU = f_Omega_SO_2p5PN(m2, m1, n21U, p2U, p1U, r12)

        self.H_SO_2p5PN = dot(Omega1_2p5PNU, S1U) + dot(Omega2_2p5PNU, S2U)

    #################################
    #################################
    def HS2011_Omega_SO_3p5PN_pt1(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> List[sp.Expr]:
        """3.5PN H_SO:  Omega_1, part 1:"""

        Omega1 = ixp.zerorank1()
        # fmt: off
        for i in range(3):
            Omega1[i] = ((+7*m2*dot(p1U,p1U)**2/(16*m1**5)
                          +9*dot(n12U,p1U)*dot(n12U,p2U)*dot(p1U,p1U)/(16*m1**4)
                          +3*dot(p1U,p1U)*dot(n12U,p2U)**2/(4*m1**3*m2)
                          +45*dot(n12U,p1U)*dot(n12U,p2U)**3/(16*m1**2*m2**2)
                          +9*dot(p1U,p1U)*dot(p1U,p2U)/(16*m1**4)
                          -3*dot(n12U,p2U)**2*dot(p1U,p2U)/(16*m1**2*m2**2)
                          -3*dot(p1U,p1U)*dot(p2U,p2U)/(16*m1**3*m2)
                          -15*dot(n12U,p1U)*dot(n12U,p2U)*dot(p2U,p2U)/(16*m1**2*m2**2)
                          +3*dot(n12U,p2U)**2*dot(p2U,p2U)/(4*m1*m2**3)
                          -3*dot(p1U,p2U)*dot(p2U,p2U)/(16*m1**2*m2**2)
                          -3*dot(p2U,p2U)**2/(16*m1*m2**3))*cross(n12U,p1U)[i])/r12**2
        # fmt: on
        return Omega1

    def HS2011_Omega_SO_3p5PN_pt2(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> List[sp.Expr]:
        """3.5PN H_SO:  Omega_1, part 2:"""
        Omega1 = ixp.zerorank1()
        # fmt: off
        for i in range(3):
            Omega1[i] = (+(-3*dot(n12U,p1U)*dot(n12U,p2U)*dot(p1U,p1U)/(2*m1**3*m2)
                           -15*dot(n12U,p1U)**2*dot(n12U,p2U)**2/(4*m1**2*m2**2)
                           +3*dot(p1U,p1U)*dot(n12U,p2U)**2/(4*m1**2*m2**2)
                           -dot(p1U,p1U)*dot(p1U,p2U)/(2*m1**3*m2)
                           +dot(p1U,p2U)**2/(2*m1**2*m2**2)
                           +3*dot(n12U,p1U)**2*dot(p2U,p2U)/(4*m1**2*m2**2)
                           -dot(p1U,p1U)*dot(p2U,p2U)/(4*m1**2*m2**2)
                           -3*dot(n12U,p1U)*dot(n12U,p2U)*dot(p2U,p2U)/(2*m1*m2**3)
                           -dot(p1U,p2U)*dot(p2U,p2U)/(2*m1*m2**3))*cross(n12U,p2U)[i])/r12**2
        # fmt: on
        return Omega1

    def HS2011_Omega_SO_3p5PN_pt3(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> List[sp.Expr]:
        """3.5PN H_SO:  Omega_1, part 3:"""
        Omega1 = ixp.zerorank1()
        # fmt: off
        for i in range(3):
            Omega1[i] = (+(-9*dot(n12U,p1U)*dot(p1U,p1U)/(16*m1**4)
                           +dot(p1U,p1U)*dot(n12U,p2U)/(m1**3*m2)
                           +27*dot(n12U,p1U)*dot(n12U,p2U)**2/(16*m1**2*m2**2)
                           -dot(n12U,p2U)*dot(p1U,p2U)/(8*m1**2*m2**2)
                           -5*dot(n12U,p1U)*dot(p2U,p2U)/(16*m1**2*m2**2))*cross(p1U,p2U)[i])/r12**2
        # fmt: on
        return Omega1

    def HS2011_Omega_SO_3p5PN_pt4(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> List[sp.Expr]:
        """3.5PN H_SO:  Omega_1, part 4:"""
        Omega1 = ixp.zerorank1()
        # fmt: off
        for i in range(3):
            Omega1[i] = (+(-3*m2*dot(n12U,p1U)**2/(2*m1**2)
                           +((-3*m2)/(2*m1**2) + 27*m2**2/(8*m1**3))*dot(p1U,p1U)
                           +(177/(16*m1) + 11/m2)*dot(n12U,p2U)**2
                           +(11/(2*m1) + 9*m2/(2*m1**2))*dot(n12U,p1U)*dot(n12U,p2U)
                           +(23/(4*m1) + 9*m2/(2*m1**2))*dot(p1U,p2U)
                           -(159/(16*m1) + 37/(8*m2))*dot(p2U,p2U))*cross(n12U,p1U)[i])/r12**3
        # fmt: on
        return Omega1

    def HS2011_Omega_SO_3p5PN_pt5(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> List[sp.Expr]:
        """3.5PN H_SO:  Omega_1, part 5:"""
        Omega1 = ixp.zerorank1()
        # fmt: off
        for i in range(3):
            Omega1[i] = (+(+4*dot(n12U,p1U)**2/m1
                           +13*dot(p1U,p1U)/(2*m1)
                           +5*dot(n12U,p2U)**2/m2
                           +53*dot(p2U,p2U)/(8*m2)
                           -(211/(8*m1) + 22/m2)*dot(n12U,p1U)*dot(n12U,p2U)
                           -(47/(8*m1) + 5/m2)*dot(p1U,p2U))*cross(n12U,p2U)[i])/r12**3
        # fmt: on
        return Omega1

    def HS2011_Omega_SO_3p5PN_pt6(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> List[sp.Expr]:
        """3.5PN H_SO:  Omega_1, part 6:"""
        Omega1 = ixp.zerorank1()
        # fmt: off
        for i in range(3):
            Omega1[i] = (+(-(8/m1 + 9*m2/(2*m1**2))*dot(n12U,p1U)
                           +(59/(4*m1) + 27/(2*m2))*dot(n12U,p2U))*cross(p1U,p2U)[i])/r12**3
        return Omega1

    def HS2011_Omega_SO_3p5PN_pt7(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        n12U: List[sp.Expr],
        p1U: List[sp.Expr],
        p2U: List[sp.Expr],
        r12: sp.Expr,
    ) -> List[sp.Expr]:
        """3.5PN H_SO:  Omega_1, part 7:"""
        Omega1 = ixp.zerorank1()
        # fmt: off
        for i in range(3):
            Omega1[i] = (+(181*m1*m2/16 + 95*m2**2/4 + 75*m2**3/(8*m1))*cross(n12U,p1U)[i]
                         -(21*m1**2/2 + 473*m1*m2/16 + 63*m2**2/4)*cross(n12U,p2U)[i])/r12**4
        # fmt: on
        return Omega1

    def f_H_SO_3p5PN(
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
        3.5PN spin-orbit coupling term, from Eq. 5 of
        Hartung and Steinhoff (2011)
        https://arxiv.org/abs/1104.3079

        3.5PN H_SO: Combining all the above Omega terms
        into the full 3.5PN S-O Hamiltonian
        expression
        """
        Omega1_3p5PNU = ixp.zerorank1()
        Omega2_3p5PNU = ixp.zerorank1()
        for i in range(3):
            Omega1_3p5PNU[i] = self.HS2011_Omega_SO_3p5PN_pt1(
                m1, m2, n12U, p1U, p2U, r12
            )[i]
            Omega1_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt2(
                m1, m2, n12U, p1U, p2U, r12
            )[i]
            Omega1_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt3(
                m1, m2, n12U, p1U, p2U, r12
            )[i]
            Omega1_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt4(
                m1, m2, n12U, p1U, p2U, r12
            )[i]
            Omega1_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt5(
                m1, m2, n12U, p1U, p2U, r12
            )[i]
            Omega1_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt6(
                m1, m2, n12U, p1U, p2U, r12
            )[i]
            Omega1_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt7(
                m1, m2, n12U, p1U, p2U, r12
            )[i]

            Omega2_3p5PNU[i] = self.HS2011_Omega_SO_3p5PN_pt1(
                m2, m1, n21U, p2U, p1U, r12
            )[i]
            Omega2_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt2(
                m2, m1, n21U, p2U, p1U, r12
            )[i]
            Omega2_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt3(
                m2, m1, n21U, p2U, p1U, r12
            )[i]
            Omega2_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt4(
                m2, m1, n21U, p2U, p1U, r12
            )[i]
            Omega2_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt5(
                m2, m1, n21U, p2U, p1U, r12
            )[i]
            Omega2_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt6(
                m2, m1, n21U, p2U, p1U, r12
            )[i]
            Omega2_3p5PNU[i] += self.HS2011_Omega_SO_3p5PN_pt7(
                m2, m1, n21U, p2U, p1U, r12
            )[i]

        self.H_SO_3p5PN = dot(Omega1_3p5PNU, S1U) + dot(Omega2_3p5PNU, S2U)


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
    in_nU: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("nU"))
    non_spinning_Hamiltonian = PN_Hamiltonian_SO(
        in_m1, in_m2, in_n12U, in_n21U, in_S1U, in_S2U, in_p1U, in_p2U, in_r12
    )
    results_dict = ve.process_dictionary_of_expressions(
        non_spinning_Hamiltonian.__dict__, fixed_mpfs_for_free_symbols=True
    )
    ve.compare_or_generate_trusted_results(
        os.path.abspath(__file__),
        os.getcwd(),
        # File basename. If this is set to "trusted_module_test1", then
        #   trusted results_dict will be stored in tests/trusted_module_test1.py
        f"{os.path.splitext(os.path.basename(__file__))[0]}",
        results_dict,
    )
