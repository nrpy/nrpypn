"""
As documented in the NRPyPN notebook
NRPyPN_shortcuts.ipynb, this Python script
provides useful shortcuts for inputting
post-Newtonian expressions into SymPy/NRPy+

Basic functions:
dot(a,b): 3-vector dot product
cross(a,b): 3-vector cross product
div(a,b): a shortcut for SymPy's sp.Rational(a,b), to declare rational numbers
num_eval(expr): Numerically evaluates NRPyPN expressions

Author:  Zachariah B. Etienne
         zachetie **at** gmail **dot* com
"""
from typing import List, cast, Optional
import sympy as sp  # SymPy: The Python computer algebra package upon which NRPy+ depends
import nrpy.indexedexp as ixp  # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support

# Step 1: Declare several global variables used
#         throughout NRPyPN
m1, m2 = sp.symbols("m1 m2", real=True)
S1U = cast(List[sp.Expr], ixp.declarerank1("S1U"))
S2U = cast(List[sp.Expr], ixp.declarerank1("S2U"))
pU = cast(List[sp.Expr], ixp.declarerank1("pU"))
nU = cast(List[sp.Expr], ixp.declarerank1("nU"))

drdt = sp.symbols("drdt", real=True)
Pt, Pr = sp.symbols("Pt Pr", real=True)
# Some references use r, others use q to represent the
#   distance between the two point masses. This is rather
#   confusing since q is also used to represent the
#   mass ratio m2/m1. However, q is the canonical position
#   variable name in Hamiltonian mechanics, so both are
#   well justified. It should be obvious which is which
#   throughout NRPyPN.
r, q = sp.symbols("r q", real=True)
chi1U = cast(List[sp.Expr], ixp.declarerank1("chi1U"))
chi2U = cast(List[sp.Expr], ixp.declarerank1("chi2U"))

# Euler-Mascheroni gamma constant:
gamma_EulerMascheroni = sp.EulerGamma

# Derived quantities used in Damour et al papers:
n12U = ixp.zerorank1()
n21U = ixp.zerorank1()
p1U = ixp.zerorank1()
p2U = ixp.zerorank1()
for _ in range(3):
    n12U[_] = +nU[_]
    n21U[_] = -nU[_]
    p1U[_] = +pU[_]
    p2U[_] = -pU[_]


# Step 2.a: Define dot and cross product of vectors
def dot(vec1: List[sp.Expr], vec2: List[sp.Expr]) -> sp.Expr:
    """
    Compute the dot product of two vectors.

    :param vec1: First vector
    :param vec2: Second vector
    :return: Dot product of vec1 and vec2
    """
    vec1_dot_vec2: sp.Expr = sp.sympify(0)
    for i in range(3):
        vec1_dot_vec2 += vec1[i] * vec2[i]
    return vec1_dot_vec2


def cross(vec1: List[sp.Expr], vec2: List[sp.Expr]) -> List[sp.Expr]:
    """
    Compute the cross product of two vectors.

    :param vec1: First vector
    :param vec2: Second vector
    :return: Cross product of vec1 and vec2
    """
    vec1_cross_vec2: List[sp.Expr] = ixp.zerorank1()
    LeviCivitaSymbol = ixp.LeviCivitaSymbol_dim3_rank3()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                vec1_cross_vec2[i] += LeviCivitaSymbol[i][j][k] * vec1[j] * vec2[k]
    return vec1_cross_vec2


def div(a: int, b: int) -> sp.Rational:
    """
    Create a SymPy Rational number a/b=Rational(a,b) from two integers a and b.

    :param a: Numerator
    :param b: Denominator
    :return: Rational number a/b
    """
    return cast(sp.Rational, sp.Rational(a, b))


# Step 3: num_eval(expr), a means to numerically evaluate SymPy/NRPyPN
#         expressions
def num_eval(
    expr: sp.Expr,
    qmassratio: float = 1.0,  # must be >= 1
    nr: float = 12.0,  # Orbital separation
    nchi1x: float = +0.0,
    nchi1y: float = +0.0,
    nchi1z: float = +0.0,
    nchi2x: float = +0.0,
    nchi2y: float = +0.0,
    nchi2z: float = +0.0,
    nPt: Optional[float] = None,
    ndrdt: Optional[float] = None,
) -> float:
    """
    Numerically evaluate SymPy/NRPyPN expressions.

    :param expr: Expression to evaluate
    :param qmassratio: Mass ratio, must be >= 1
    :param nr: Orbital separation
    :param nchi1x: x-component of the first spin
    :param nchi1y: y-component of the first spin
    :param nchi1z: z-component of the first spin
    :param nchi2x: x-component of the second spin
    :param nchi2y: y-component of the second spin
    :param nchi2z: z-component of the second spin
    :param nPt: Optional Pt value
    :param ndrdt: Optional drdt value
    :return: Numerical value of the expression
    """
    # DERIVED QUANTITIES BELOW
    # We want m1+m2 = 1, so that
    #         m2/m1 = qmassratio
    # We find below:
    nm1 = 1 / (1 + qmassratio)
    nm2 = qmassratio / (1 + qmassratio)
    # This way nm1+nm2 = (qmassratio+1)/(1+qmassratio) = 1 CHECK
    #      and nm2/nm1 = qmassratio                        CHECK

    nS1U0 = nchi1x * nm1**2
    nS1U1 = nchi1y * nm1**2
    nS1U2 = nchi1z * nm1**2
    nS2U0 = nchi2x * nm2**2
    nS2U1 = nchi2y * nm2**2
    nS2U2 = nchi2z * nm2**2

    if nPt:
        expr2 = expr.subs(Pt, nPt)
        expr = expr2
    if ndrdt:
        expr2 = expr.subs(drdt, ndrdt)
        expr = expr2
    return cast(
        float,
        expr.subs(m1, nm1)
        .subs(m2, nm2)
        .subs(S1U[0], nS1U0)
        .subs(S1U[1], nS1U1)
        .subs(S1U[2], nS1U2)
        .subs(S2U[0], nS2U0)
        .subs(S2U[1], nS2U1)
        .subs(S2U[2], nS2U2)
        .subs(chi1U[0], nchi1x)
        .subs(chi1U[1], nchi1y)
        .subs(chi1U[2], nchi1z)
        .subs(chi2U[0], nchi2x)
        .subs(chi2U[1], nchi2y)
        .subs(chi2U[2], nchi2z)
        .subs(r, nr)
        .subs(q, nr)
        .subs(sp.pi, sp.N(sp.pi))
        .subs(gamma_EulerMascheroni, sp.N(sp.EulerGamma)),
    )
