from parameters import parameters as p
from run_construction_domain import cd_global, cd_difference, cd_dpp
from CFL.cfl import rover_cdpp, rover_cdif


if __name__ == '__main__':
    """
    Run construction domain using either G, D, D++, or CFL
    This file trains rover policies directly (not skills based)
    """

    assert (p["algorithm"] != "CKI" and p["algorithm"] != "ACG")

    if p["algorithm"] == "Global":
        print("Construction Domain: Global Rewards")
        cd_global()
    elif p["algorithm"] == "Difference":
        print("Construction Domain: Difference Rewards")
        cd_difference()
    elif p["algorithm"] == "DPP":
        print("Construction Domain: D++ Rewards")
        cd_dpp()
    elif p["algorithm"] == "CFL":
        print("Construction Domain: CFL")
        rover_cdpp()
    else:
        print("ALGORITHM TYPE ERROR")
