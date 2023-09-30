"""
As documented in the NRPyPN notebook
NRPyPN_shortcuts.ipynb, this Python script
provides useful shortcuts for inputting
post-Newtonian expressions into SymPy/NRPy+

Basic functions:

Author:  Zachariah B. Etienne
         zachetie **at** gmail **dot* com
"""
from typing import Tuple, Dict
from collections import namedtuple

from nrpypn.NRPyPN_shortcuts import (
    m1,
    m2,
    chi1U,
    chi2U,
    r,
    n12U,
    n21U,
    S1U,
    S2U,
    p1U,
    p2U,
    num_eval,
)
from nrpypn.PN_p_t import PN_p_t
from nrpypn.PN_p_r import PN_p_r


def eval__P_t__and__P_r(
    qmassratio: float,
    nr: float,
    nchi1x: float,
    nchi1y: float,
    nchi1z: float,
    nchi2x: float,
    nchi2y: float,
    nchi2z: float,
) -> Tuple[float, float]:
    """
    Core function to numerically evaluate P_t and P_r

    :param qmassratio: mass ratio q >= 1
    :param nr: Orbital separation (total coordinate distance between black holes)
    :param nchi1x: x-component of the dimensionless spin vector for BH 1
    :param nchi1y: y-component of the dimensionless spin vector for BH 1
    :param nchi1z: z-component of the dimensionless spin vector for BH 1
    :param nchi2x: x-component of the dimensionless spin vector for BH 2
    :param nchi2y: y-component of the dimensionless spin vector for BH 2
    :param nchi2z: z-component of the dimensionless spin vector for BH 2
    :return: Tuple containing the numerical values for P_t (the tangential momentum) and P_r (the radial momentum)
    """
    # Compute p_t, the tangential component of momentum
    pt = PN_p_t(m1, m2, chi1U, chi2U, r)

    # Compute p_r, the radial component of momentum
    pr = PN_p_r(m1, m2, n12U, n21U, chi1U, chi2U, S1U, S2U, p1U, p2U, r)

    nPt = num_eval(
        pt.p_t,
        qmassratio=qmassratio,
        nr=nr,
        nchi1x=nchi1x,
        nchi1y=nchi1y,
        nchi1z=nchi1z,
        nchi2x=nchi2x,
        nchi2y=nchi2y,
        nchi2z=nchi2z,
    )

    nPr = num_eval(
        pr.p_r,
        qmassratio=qmassratio,
        nr=nr,
        nchi1x=nchi1x,
        nchi1y=nchi1y,
        nchi1z=nchi1z,
        nchi2x=nchi2x,
        nchi2y=nchi2y,
        nchi2z=nchi2z,
        nPt=nPt,
    )
    return float(nPt), float(nPr)


def test_results_against_trusted() -> None:
    """
    Test the function eval__P_t__and__P_r against trusted values.

    :raises ValueError: If the relative error exceeds the tolerance for either p_t or p_r.
    """
    bbh_params = namedtuple(
        "bbh_params",
        [
            "q",
            "d",
            "chi1x",
            "chi1y",
            "chi1z",
            "chi2x",
            "chi2y",
            "chi2z",
            "tr_p_t",
            "tr_p_r",
        ],
    )
    case_dict: Dict[str, bbh_params] = {
        "Mass ratio q=2, chi1= (0,0,0); chi2=(-0.3535, 0.3535, 0.5), radial separation r=10.8": bbh_params(
            q=2.0,
            d=10.8,
            chi1x=0,
            chi1y=0,
            chi1z=0,
            chi2x=-0.3535,
            chi2y=0.3535,
            chi2z=0.5,
            tr_p_t=0.0793500403866190,
            tr_p_r=0.0005426257166677216,
        ),
        "Mass ratio q=8, chi1= (0, 0, 0.5); chi2=(0, 0, 0.5), radial separation r=11": bbh_params(
            q=8.0,
            d=11.0,
            chi1x=0,
            chi1y=0,
            chi1z=0.5,
            chi2x=0,
            chi2y=0,
            chi2z=0.5,
            tr_p_t=0.0345503689803291,
            tr_p_r=0.0000975735300199125,
        ),
        "Mass ratio q=1.5, chi1= (0,0,-0.6); chi2=(0,0,0.6), radial separation r=10.8": bbh_params(
            q=1.5,
            d=10.8,
            chi1x=0,
            chi1y=0,
            chi1z=-0.6,
            chi2x=0,
            chi2y=0,
            chi2z=0.6,
            tr_p_t=0.0868556558764586,
            tr_p_r=0.0006770196671656516,
        ),
        "Mass ratio q=4, chi1= (0,0,-0.8); chi2=(0,0,0.8), radial separation r=11": bbh_params(
            q=4,
            d=11,
            chi1x=0,
            chi1y=0,
            chi1z=-0.8,
            chi2x=0,
            chi2y=0,
            chi2z=0.8,
            tr_p_t=0.0558077537453816,
            tr_p_r=0.0002511426409814753,
        ),
        "Mass ratio q=1, chi1=chi2=(0,0,0), radial separation r=12": bbh_params(
            q=1,
            d=12,
            chi1x=0,
            chi1y=0,
            chi1z=0,
            chi2x=0,
            chi2y=0,
            chi2z=0,
            tr_p_t=0.0850940927209620,
            tr_p_r=0.0005398602170955123,
        ),
        "Mass ratio q=1.391, chi1=(0.4381,-0.221,0.314), chi2=(-0.1,0.4,-0.26), radial separation r=8.124981": bbh_params(
            q=1.391,
            d=8.124981,
            chi1x=0.4381,
            chi1y=-0.221,
            chi1z=0.314,
            chi2x=-0.1,
            chi2y=0.4,
            chi2z=-0.26,
            tr_p_t=0.10894035623063517,
            tr_p_r=0.002093970161808933,
        ),
    }

    for key, value in sorted(case_dict.items()):
        p_t, p_r = eval__P_t__and__P_r(
            qmassratio=value.q,
            nr=value.d,
            nchi1x=value.chi1x,
            nchi1y=value.chi1y,
            nchi1z=value.chi1z,
            nchi2x=value.chi2x,
            nchi2y=value.chi2y,
            nchi2z=value.chi2z,
        )
        # rel_error = abs(value.tr_p_t - p_t) / value.tr_p_t < 1e-13 -> abs(value.tr_p_t - p_t) < 1e-13 * value.tr_p_t
        rel_error_p_t = abs(value.tr_p_t - p_t) / value.tr_p_t
        rel_error_p_r = abs(value.tr_p_r - p_r) / value.tr_p_r
        tol = 1.0e-13
        if rel_error_p_t > tol:
            raise ValueError(
                f"Error: p_t relative error ({rel_error_p_t}) too large, when compared with trusted value"
            )
        if rel_error_p_r > tol:
            raise ValueError(
                f"Error: p_r relative error ({rel_error_p_r}) too large, when compared with trusted value"
            )
        print(value.tr_p_t, value.tr_p_r, key)
        print(p_t, p_r, key)
        print(rel_error_p_t, rel_error_p_r)
