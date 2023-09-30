"""
Driver Python script for generating quasicircular
parameters, accurate to 3.5PN, for tangential
and radial momenta for e.g., binary black holes.

Author:  Zachariah B. Etienne
         zachetie **at** gmail **dot* com
"""

import argparse
from nrpypn.eval_p_t_and_p_r import eval__P_t__and__P_r


def main() -> None:
    """
    Main function for evaluating P_t and P_r for in quasi-circular orbits.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate P_t and P_r for e.g., black holes in quasi-circular orbits."
    )
    parser.add_argument("qmassratio", type=float, nargs="?", help="Mass ratio q >= 1")
    parser.add_argument(
        "nr",
        type=float,
        nargs="?",
        help="Orbital separation (total coordinate distance between black holes)",
    )
    parser.add_argument(
        "nchi1x",
        type=float,
        nargs="?",
        help="x-component of the dimensionless spin vector for BH 1",
    )
    parser.add_argument(
        "nchi1y",
        type=float,
        nargs="?",
        help="y-component of the dimensionless spin vector for BH 1",
    )
    parser.add_argument(
        "nchi1z",
        type=float,
        nargs="?",
        help="z-component of the dimensionless spin vector for BH 1",
    )
    parser.add_argument(
        "nchi2x",
        type=float,
        nargs="?",
        help="x-component of the dimensionless spin vector for BH 2",
    )
    parser.add_argument(
        "nchi2y",
        type=float,
        nargs="?",
        help="y-component of the dimensionless spin vector for BH 2",
    )
    parser.add_argument(
        "nchi2z",
        type=float,
        nargs="?",
        help="z-component of the dimensionless spin vector for BH 2",
    )

    args = parser.parse_args()

    # Check for missing arguments and raise appropriate errors
    if any(arg is None for arg in vars(args).values()):
        print(
            "****************************************************************************"
            "\nThis program evaluates P_t and P_r for black holes in quasi-circular orbits."
        )
        print("It assumes the black holes are:")
        print(
            "* Situated initially on the x-axis, with black hole 1 at x>0, and black hole 2 at x<0."
        )
        print(
            "* Orbiting instantaneously in the x-y plane, with the center-of-mass at x=y=z=0."
        )
        print(
            "If you plan to use these parameters with TwoPunctures, you will need to set:"
        )
        print("TwoPunctures::give_bare_mass = no")
        print("TwoPunctures::target_m_minus = q/(1+q)")
        print("TwoPunctures::target_m_plus = 1/(1+q)")
        print("TwoPunctures::par_P_plus[0] = -|P_r| (from output)")
        print("TwoPunctures::par_P_plus[1] = +|P_t| (from output)")
        print("TwoPunctures::par_P_minus[0] = +|P_r| (from output)")
        print("TwoPunctures::par_P_minus[1] = -|P_t| (from output)")
        print(
            "TwoPunctures::par_S_plus[0] = nchi1x*TwoPunctures::target_m_plus*TwoPunctures::target_m_plus (from input into code cell below)"
        )
        print(
            "TwoPunctures::par_S_plus[1] = nchi1y*TwoPunctures::target_m_plus*TwoPunctures::target_m_plus (from input into code cell below)"
        )
        print(
            "TwoPunctures::par_S_plus[2] = nchi1z*TwoPunctures::target_m_plus*TwoPunctures::target_m_plus (from input into code cell below)"
        )
        print("... similarly for TwoPunctures::par_S_minus[]\n")
        parser.print_help()
    else:
        pt, pr = eval__P_t__and__P_r(
            args.qmassratio,
            args.nr,
            args.nchi1x,
            args.nchi1y,
            args.nchi1z,
            args.nchi2x,
            args.nchi2y,
            args.nchi2z,
        )
        print(f"P_t: {pt}, P_r: {pr}")


if __name__ == "__main__":
    main()
