"""
As documented in the NRPyPN notebook
PN-Hamiltonian-Nonspinning.ipynb, this Python script
generates nonspinning pieces of the post-Newtonian (PN)
Hamiltonian, up to and including third PN order.

Author:  Zachariah B. Etienne
         zachetie **at** gmail **dot* com
"""

from typing import List, cast
import sympy as sp  # SymPy: The Python computer algebra package upon which NRPy+ depends
import nrpy.indexedexp as ixp  # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import nrpy.validate_expressions.validate_expressions as ve

# NRPyPN: shortcuts for e.g., vector operations
from nrpypn.NRPyPN_shortcuts import div, dot


class PN_Hamiltonian_NS:
    """
    Class to handle Hamiltonian calculations in Newtonian and Post-Newtonian (PN) formalism for non-spinning objects.

    :param m1: Mass of the first object
    :param m2: Mass of the second object
    :param PU: Momentum 3-vector
    :param nU: Unit vector pointing from m1 to m2
    :param q: Scalar related to the separation between m1 and m2

    Attributes:
    - H_Newt (sp.Expr): Newtonian Hamiltonian
    - H_NS_1PN (sp.Expr): 1st order Post-Newtonian Hamiltonian
    - H_NS_2PN (sp.Expr): 2nd order Post-Newtonian Hamiltonian
    - H_NS_3PN (sp.Expr): 3rd order Post-Newtonian Hamiltonian

    Basic functions:
    f_H_Newt__H_NS_1PN__H_NS_2PN(m1,m2, pU, nU, q): Compute H_Newt,
                                                    H_NS_1PN, and H_NS_2PN
                                                    and store to class
                                                    variables of the same
                                                    names.
    f_H_NS_3PN: Compute H_NS_3PN, and store to global variable of same name
    """

    def __init__(
        self, m1: sp.Expr, m2: sp.Expr, PU: List[sp.Expr], nU: List[sp.Expr], q: sp.Expr
    ) -> None:
        self.H_Newt: sp.Expr = sp.sympify(0)
        self.H_NS_1PN: sp.Expr = sp.sympify(0)
        self.H_NS_2PN: sp.Expr = sp.sympify(0)
        self.H_NS_3PN: sp.Expr = sp.sympify(0)

        self.f_H_Newt__H_NS_1PN__H_NS_2PN(m1, m2, PU, nU, q)
        self.f_H_NS_3PN(m1, m2, PU, nU, q)

    def f_H_Newt__H_NS_1PN__H_NS_2PN(
        self, m1: sp.Expr, m2: sp.Expr, PU: List[sp.Expr], nU: List[sp.Expr], q: sp.Expr
    ) -> None:
        """
        Compute H_Newt, H_NS_1PN, and H_NS_2PN and store them in the class variables of the same names.
        """
        mu = m1 * m2 / (m1 + m2)
        eta = m1 * m2 / (m1 + m2) ** 2
        pU = ixp.zerorank1()
        for i in range(3):
            pU[i] = PU[i] / mu

        self.H_Newt = mu * (+div(1, 2) * dot(pU, pU) - 1 / q)

        # fmt: off
        self.H_NS_1PN = mu*(+div(1,8)*(3*eta-1)*dot(pU,pU)**2
                       -div(1,2)*((3+eta)*dot(pU,pU) + eta*dot(nU,pU)**2)/q
                       +div(1,2)/q**2)

        self.H_NS_2PN = mu*(+div(1,16)*(1 -  5*eta + 5*eta**2)*dot(pU,pU)**3
                       +div(1,8)*(+(5 - 20*eta - 3*eta**2)*dot(pU,pU)**2
                                  -2*eta**2*dot(nU,pU)**2*dot(pU,pU)
                                  -3*eta**2*dot(nU,pU)**4)/q
                       +div(1,2)*((5+8*eta)*dot(pU,pU) + 3*eta*dot(nU,pU)**2)/q**2
                       -div(1,4)*(1+3*eta)/q**3)
        # fmt: on

    def f_H_NS_3PN(
        self, m1: sp.Expr, m2: sp.Expr, PU: List[sp.Expr], nU: List[sp.Expr], q: sp.Expr
    ) -> None:
        """
        Compute H_NS_3PN and store it in the class variable of the same name.
        """
        mu = m1 * m2 / (m1 + m2)
        eta = m1 * m2 / (m1 + m2) ** 2
        pU = ixp.zerorank1()
        for i in range(3):
            pU[i] = PU[i] / mu

        # fmt: off
        self.H_NS_3PN = mu*(+div(1,128)*(-5 + 35*eta - 70*eta**2 + 35*eta**3)*dot(pU,pU)**4
                       +div(1, 16)*(+(-7 + 42*eta - 53*eta**2 -  5*eta**3)*dot(pU,pU)**3
                                    +(2-3*eta)*eta**2*dot(nU,pU)**2*dot(pU,pU)**2
                                    +3*(1-eta)*eta**2*dot(nU,pU)**4*dot(pU,pU) - 5*eta**3*dot(nU,pU)**6)/q
                       +(+div(1,16)*(-27 + 136*eta + 109*eta**2)*dot(pU,pU)**2
                         +div(1,16)*(+17 +  30*eta)*eta*dot(nU,pU)**2*dot(pU,pU)
                         +div(1,12)*(+ 5 +  43*eta)*eta*dot(nU,pU)**4)/q**2
                       +(+(-div(25, 8) + (div(1,64)*sp.pi**2 - div(335,48))*eta - div(23,8)*eta**2)*dot(pU,pU)
                         +(-div(85,16) - div(3,64)*sp.pi**2 - div(7,4)*eta)*eta*dot(nU,pU)**2)/q**3
                       +(+div(1,8)+(div(109,12) - div(21,32)*sp.pi**2)*eta)/q**4)
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

    in_m1, in_m2, in_q = sp.symbols("m1 m2 q")
    in_PU: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("PU"))
    in_nU: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("nU"))
    non_spinning_Hamiltonian = PN_Hamiltonian_NS(in_m1, in_m2, in_PU, in_nU, in_q)
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
