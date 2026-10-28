import scipy.constants as c
from MSN_project12.old.utils_old import DoubleWellPotential, Trajectory, RingPolymerLangevin
kb = c.k
hbar = c.hbar

# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------
def main(n_steps=50000, P=200, gamma=None, alpha=1/100, seed=None, show=True):
    a = 0.04e-9
    C = 0.8
    mass = c.m_p

    V0 = hbar**2 / (2 * mass * a**2 * C)
    pot = DoubleWellPotential(V0, a)
    T = 0.2 * pot.V0 / kb

    print(f"V_0 = {pot.V0:.2e} J = {pot.V0 / c.e:.2f} eV")
    print(f"T = {T:.2e} K")

    sim = RingPolymerLangevin(pot, mass, T, P=P, xc = None, gamma=gamma, seed=seed, alpha = alpha)
    print(f"dt = {sim.dt:.3e} s, P = {P}")

    traj = sim.run(n_steps)
    sim.plot(traj, filename="trajectoires1.png", show=show)
    return sim, traj


if __name__ == "__main__":
    """
    Ci-dessous, on reponds aux differentes questions du projet
    """
    # --- Etape 1 ---
    """
    On verifie le cas classique (P=1, gamma=0)
    """
    main(n_steps=10000, P=16, gamma=None, alpha=1/100, seed=None, show=True)
