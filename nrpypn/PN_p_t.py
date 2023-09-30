"""
As documented in the NRPyPN notebook
PN-p_t.ipynb, this Python script
generates the expression for the transverse
component of momentum p_t up to and
including terms at 3.5PN order.
This is an implementation of the equations of
 Ramos-Buades, Husa, and Pratten (2018)
   https://arxiv.org/abs/1810.00036
but validates against the relevant equation
  in Healy, Lousto, Nakano, and Zlochower (2017)
   https://arxiv.org/abs/1702.00872

Author:  Zachariah B. Etienne
         zachetie **at** gmail **dot* com
"""
from typing import List, cast
import sympy as sp  # SymPy: The Python computer algebra package upon which NRPy+ depends
import nrpy.indexedexp as ixp  # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import nrpy.validate_expressions.validate_expressions as ve

from nrpypn.NRPyPN_shortcuts import div  # NRPyPN: shortcuts for e.g., vector operations


class PN_p_t:
    """
    Basic equation for p_t can be written in the
    form:
    p_t = q / (sqrt(r) * (1 + q)^2) * (1 + sum_{k=2}^7 (a_k / r^{k/2}))
    where we construct the a_k terms in the sum below.

    :param m1: mass of object 1
    :param m2: mass of object 2
    :param chi1U: spin of object 1
    :param chi2U: spin of object 2
    :param r: radius of separation
    """

    def __init__(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        chi1U: List[sp.Expr],
        chi2U: List[sp.Expr],
        r: sp.Expr,
    ):
        self.a_2: sp.Expr = sp.sympify(0)
        self.a_3: sp.Expr = sp.sympify(0)
        self.a_4: sp.Expr = sp.sympify(0)
        self.a_5: sp.Expr = sp.sympify(0)
        self.a_6: sp.Expr = sp.sympify(0)
        self.a_7: sp.Expr = sp.sympify(0)

        self.p_t: sp.Expr = sp.sympify(0)

        self.f_p_t(m1, m2, chi1U, chi2U, r)

    #################################
    #################################
    def p_t__a_2_thru_a_4(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        chi1x: sp.Expr,
        chi1y: sp.Expr,
        chi1z: sp.Expr,
        chi2x: sp.Expr,
        chi2y: sp.Expr,
        chi2z: sp.Expr,
    ) -> None:
        """
        Step 1: Construct terms a_2, a_3, and a_4, from
        Eq A2 of Ramos-Buades, Husa, and Pratten (2018)
        https://arxiv.org/abs/1810.00036
        These terms have been independently validated
        against the same terms in Eq 7 of
        Healy, Lousto, Nakano, and Zlochower (2017)
        https://arxiv.org/abs/1702.00872

        :param m1: mass of object 1
        :param m2: mass of object 2
        :param chi1x: x-component of spin of object 1
        :param chi1y: y-component of spin of object 1
        :param chi1z: z-component of spin of object 1
        :param chi2x: x-component of spin of object 2
        :param chi2y: y-component of spin of object 2
        :param chi2z: z-component of spin of object 2
        """
        # fmt: off
        q = m2/m1 # It is assumed that q >= 1, so m2 >= m1.
        self.a_2 = sp.sympify(2)
        self.a_3 = (-3*(4*q**2+3*q)*chi2z/(4*(q+1)**2) - 3*(3*q+4)*chi1z/(4*(q+1)**2))
        self.a_4 = (-3*q**2*chi2x**2/(2*(q+1)**2)
                    +3*q**2*chi2y**2/(4*(q+1)**2)
                    +3*q**2*chi2z**2/(4*(q+1)**2)
                    +(+42*q**2 + 41*q + 42)/(16*(q+1)**2)
                    -3*chi1x**2/(2*(q+1)**2)
                    -3*q*chi1x*chi2x/(q+1)**2
                    +3*chi1y**2/(4*(q+1)**2)
                    +3*q*chi1y*chi2y/(2*(q+1)**2)
                    +3*chi1z**2/(4*(q+1)**2)
                    +3*q*chi1z*chi2z/(2*(q+1)**2))
        # fmt: on

    def p_t__a_5_thru_a_6(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        chi1x: sp.Expr,
        chi1y: sp.Expr,
        chi1z: sp.Expr,
        chi2x: sp.Expr,
        chi2y: sp.Expr,
        chi2z: sp.Expr,
        FixSignError: bool = True,
    ) -> None:
        """
        Construct terms a_5 and a_6, from
        Eq A2 of Ramos-Buades, Husa, and Pratten (2018)
        https://arxiv.org/abs/1810.00036
        These terms have been independently validated
        against the same terms in Eq 7 of
        Healy, Lousto, Nakano, and Zlochower (2017)
        https://arxiv.org/abs/1702.00872
        and a sign error was corrected in the a_5
        expression.

        :param m1: mass of object 1
        :param m2: mass of object 2
        :param chi1x: x-component of spin of object 1
        :param chi1y: y-component of spin of object 1
        :param chi1z: z-component of spin of object 1
        :param chi2x: x-component of spin of object 2
        :param chi2y: y-component of spin of object 2
        :param chi2z: z-component of spin of object 2
        :param FixSignError: boolean to determine if sign error should be fixed
        """
        # Initialize the SignFix based on FixSignError
        SignFix = sp.sympify(-1) if FixSignError else sp.sympify(1)

        # It is assumed that q >= 1, so m2 >= m1.
        q = m2 / m1
        # fmt: off
        self.a_5 = (SignFix*(13*q**3 + 60*q**2 + 116*q + 72)*chi1z/(16*(q+1)**4)
                    +(-72*q**4 - 116*q**3 - 60*q**2 - 13*q)*chi2z/(16*(q+1)**4))
        self.a_6 = (+(+472*q**2 - 640)*chi1x**2/(128*(q+1)**4)
                    +(-512*q**2 - 640*q - 64)*chi1y**2/(128*(q+1)**4)
                    +(-108*q**2 + 224*q +512)*chi1z**2/(128*(q+1)**4)
                    +(+472*q**2 - 640*q**4)*chi2x**2/(128*(q+1)**4)
                    +(+192*q**3 + 560*q**2 + 192*q)*chi1x*chi2x/(128*(q+1)**4)
                    +(-864*q**3 -1856*q**2 - 864*q)*chi1y*chi2y/(128*(q+1)**4)
                    +(+480*q**3 +1064*q**2 + 480*q)*chi1z*chi2z/(128*(q+1)**4)
                    +( -64*q**4 - 640*q**3 - 512*q**2)*chi2y**2/(128*(q+1)**4)
                    +(+512*q**4 + 224*q**3 - 108*q**2)*chi2z**2/(128*(q+1)**4)
                    +(+480*q**4 + 163*sp.pi**2*q**3 - 2636*q**3 + 326*sp.pi**2*q**2 - 6128*q**2 + 163*sp.pi**2*q-2636*q+480)
                     /(128*(q+1)**4))
        # fmt: on

    def p_t__a_7(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        chi1x: sp.Expr,
        chi1y: sp.Expr,
        chi1z: sp.Expr,
        chi2x: sp.Expr,
        chi2y: sp.Expr,
        chi2z: sp.Expr,
    ) -> None:
        """
        Construct term a_7, from Eq A2 of
         Ramos-Buades, Husa, and Pratten (2018)
           https://arxiv.org/abs/1810.00036

        :param m1: mass of object 1
        :param m2: mass of object 2
        :param chi1x: x-component of spin of object 1
        :param chi1y: y-component of spin of object 1
        :param chi1z: z-component of spin of object 1
        :param chi2x: x-component of spin of object 2
        :param chi2y: y-component of spin of object 2
        :param chi2z: z-component of spin of object 2
        """
        q = m2 / m1  # It is assumed that q >= 1, so m2 >= m1.
        # fmt: off
        self.a_7 = (+5*(4*q+1)*q**3*chi2x**2*chi2z/(2*(q+1)**4)
                    -5*(4*q+1)*q**3*chi2y**2*chi2z/(8*(q+1)**4)
                    -5*(4*q+1)*q**3*chi2z**3      /(8*(q+1)**4)
                    +chi1x*(+15*(2*q+1)*q**2*chi2x*chi2z/(4*(q+1)**4)
                            +15*(1*q+2)*q   *chi2x*chi1z/(4*(q+1)**4))
                    +chi1y*(+15*q**2*chi2y*chi1z/(4*(q+1)**4)
                            +15*q**2*chi2y*chi2z/(4*(q+1)**4))
                    +chi1z*(+15*q**2*(2*q+3)*chi2x**2/(4*(q+1)**4)
                            -15*q**2*(  q+2)*chi2y**2/(4*(q+1)**4)
                            -15*q**2        *chi2z**2/(4*(q+1)**3)
                            -(103*q**5 + 145*q**4 - 27*q**3 + 252*q**2 + 670*q + 348)/(32*(q+1)**6))
                    -(+348*q**5 + 670*q**4 + 252*q**3 - 27*q**2 + 145*q + 103)*q*chi2z/(32*(q+1)**6)
                    +chi1x**2*(+5*(q+4)*chi1z/(2*(q+1)**4)
                               +15*q*(3*q+2)*chi2z/(4*(q+1)**4))
                    +chi1y**2*(-5*(q+4)*chi1z/(8*(q+1)**4)
                               -15*q*(2*q+1)*chi2z/(4*(q+1)**4))
                    -15*q*chi1z**2*chi2z/(4*(q+1)**3)
                    -5*(q+4)*chi1z**3/(8*(q+1)**4))
        # fmt: on

    def f_p_t(
        self,
        m1: sp.Expr,
        m2: sp.Expr,
        chi1U: List[sp.Expr],
        chi2U: List[sp.Expr],
        r: sp.Expr,
    ) -> None:
        """
        Finally, sum the expressions for a_k to construct p_t as prescribed:
        p_t = q/(sqrt(r)*(1+q)^2) (1 + sum_{k=2}^7 (a_k/r^{k/2}))

        :param m1: mass of object 1
        :param m2: mass of object 2
        :param chi1U: spin of object 1
        :param chi2U: spin of object 2
        :param r: radius of separation
        """
        q = m2 / m1  # It is assumed that q >= 1, so m2 >= m1.
        a = ixp.zerorank1(dimension=10)
        self.p_t__a_2_thru_a_4(
            m1, m2, chi1U[0], chi1U[1], chi1U[2], chi2U[0], chi2U[1], chi2U[2]
        )
        self.p_t__a_5_thru_a_6(
            m1, m2, chi1U[0], chi1U[1], chi1U[2], chi2U[0], chi2U[1], chi2U[2]
        )
        self.p_t__a_7(
            m1, m2, chi1U[0], chi1U[1], chi1U[2], chi2U[0], chi2U[1], chi2U[2]
        )
        a[2] = self.a_2
        a[3] = self.a_3
        a[4] = self.a_4
        a[5] = self.a_5
        a[6] = self.a_6
        a[7] = self.a_7
        self.p_t = sp.sympify(1)  # Term prior to the sum in parentheses
        for k in range(8):
            self.p_t += a[k] / r ** div(k, 2)
        self.p_t *= q / (1 + q) ** 2 * 1 / r ** div(1, 2)


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

    in_m1, in_m2, in_r = sp.symbols("m1 m2, r")
    in_chi1U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("chi1U"))
    in_chi2U: List[sp.Expr] = cast(List[sp.Expr], ixp.declarerank1("chi2U"))
    tangential_momentum = PN_p_t(in_m1, in_m2, in_chi1U, in_chi2U, in_r)
    results_dict = ve.process_dictionary_of_expressions(
        tangential_momentum.__dict__, fixed_mpfs_for_free_symbols=True
    )
    ve.compare_or_generate_trusted_results(
        os.path.abspath(__file__),
        os.getcwd(),
        # File basename. If this is set to "trusted_module_test1", then
        #   trusted results_dict will be stored in tests/trusted_module_test1.py
        f"{os.path.splitext(os.path.basename(__file__))[0]}",
        results_dict,
    )
