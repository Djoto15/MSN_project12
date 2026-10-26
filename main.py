"""Dynamique de Langevin d'un polymère en anneau (PIMD) dans un double puits.

Version orientée objet et vectorisée de main_old.py.
Convention conservée : les P répliques évoluent à la température effective
P*T, avec ressorts de pulsation w_p = P kB T / hbar et potentiel V non divisé
par P (équivalent à la formulation avec V/P à la température T).
"""
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c

kb = c.k
hbar = c.hbar


class DoubleWellPotential:
    """V(x) = V0 ((x/a)^2 - 1)^2"""

    def __init__(self, V0, a):
        self.V0 = V0
        self.a = a

    def energy(self, x):
        return self.V0 * ((x / self.a) ** 2 - 1) ** 2

    def gradient(self, x):
        return 4 * self.V0 * x / self.a**2 * ((x / self.a) ** 2 - 1)



@dataclass
class Trajectory:
    t: np.ndarray   # (n_steps+1,)
    X: np.ndarray   # (n_steps+1, P) positions
    V: np.ndarray   # (n_steps+1, P) vitesses



class RingPolymerLangevin:
    def __init__(self, potential, mass, T, P=1, gamma=None, alpha=1/100,
                 seed=None):
        self.pot = potential
        self.m = mass
        self.T = T
        self.P = P
        self.rng = np.random.default_rng(seed)  
        self.w_p = P * kb * T / hbar
        self.w0 = np.sqrt(4 * potential.V0 / (mass * potential.a**2))
        self.gamma = 0.1 * self.w_p/(2*np.pi) if gamma is None else gamma
        self.dt = alpha * 2 * np.pi / max(self.w_p, self.w0)

        # Constantes pré-calculées
        self._noise_amp = np.sqrt(2 * mass * self.gamma * kb * P * T / self.dt)
        self._k_spring = mass * self.w_p**2


    # --- Configurqtion initiale ---
    def vitess_init(self):
        v_0 = np.sqrt(kb * self.P * self.T / self.m)
        return self.rng.normal(0.0, v_0, size=self.P)   # # tire P vitesses initiales selon une loi normale centrée et de sigma v_0

    def pos_init(self):
        return np.zeros(self.P)

    # --- Forces ---
    def spring_force(self, x):
        return -self._k_spring * (2 * x - np.roll(x, 1) - np.roll(x, -1))

    def force(self, x, v, noise):
        """Force totale sur les P répliques. `noise` ~ N(0,1) de taille P."""
        return (-self.pot.gradient(x)
                + self.spring_force(x)
                + self._noise_amp * noise
                - self.gamma * self.m * v)

    # --- Integration ---
    def run(self, n_steps):
        """Verlet (positions) + vitesse par différence finie arrière d'ordre 2."""
        P, m, dt = self.P, self.m, self.dt
        X = np.empty((n_steps + 1, P))
        V = np.empty((n_steps + 1, P))
        X[0] = self.pos_init()
        V[0] = self.vitess_init()

        noise = self.rng.standard_normal((n_steps, P))  # tiré en une fois

        # Premier pas : développement de Taylor
        F = self.force(X[0], V[0], noise[0])
        X[1] = X[0] + V[0] * dt + dt**2 / (2 * m) * F
        V[1] = V[0] + F * dt / m

        dt2_m = dt**2 / m
        inv_2dt = 1.0 / (2 * dt)
        for t in range(1, n_steps):
            F = self.force(X[t], V[t], noise[t])
            X[t + 1] = 2 * X[t] - X[t - 1] + dt2_m * F
            V[t + 1] = (3 * X[t + 1] - 4 * X[t] + X[t - 1]) * inv_2dt

        return Trajectory(t=np.arange(n_steps + 1) * dt, X=X, V=V)

    # --- observables conformes à la slide 30 du cours MSN3 ----------------
    def classical_kinetic_energy(self, traj):
        """1er terme de l'estimateur : (1/P) sum_s 1/2 m v_s^2 (car v_s est à T_eff = P*T)."""
        return 0.5 * self.m * np.mean(traj.V**2, axis=1)

    def spring_energy(self, traj):
        """2e terme de l'estimateur : (1/P) sum_s 1/2 m w_p^2 (x_s - x_{s+1})^2."""
        dX = traj.X - np.roll(traj.X, -1, axis=1)
        return 0.5 * self._k_spring * np.mean(dX**2, axis=1)
    
    def kinetic_energy(self, traj):
        """Estimateur de l'énergie cinétique quantique (Slide 30)."""
        return self.classical_kinetic_energy(traj) - self.spring_energy(traj)

    def potential_energy(self, traj):
        """Énergie potentielle du système (Slide 30) : (1/P) sum_s V(x_s)."""
        return np.mean(self.pot.energy(traj.X), axis=1)

    @staticmethod
    def centroid(traj):
        return traj.X.mean(axis=1)

    
    
    
    # --- sauvegarde / tracés ---------------------------------------------
    def save_csv(self, traj, filename="resultats.csv"):
        P = self.P
        header = ",".join(["t_s"]
                          + [f"x_bead{s}_m" for s in range(P)]
                          + [f"v_bead{s}_m_s" for s in range(P)])
        data = np.column_stack([traj.t, traj.X, traj.V])
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        return filename

    def plot(self, traj, filename=None, show=True):
        t = traj.t
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(t, self.classical_kinetic_energy(traj)/(0.5*kb*self.P), label="Cinetique classique", lw=0.8)
        axes[0].plot(t, self.kinetic_energy(traj)/(0.5*kb*self.P), label="Cinetique Quantique", lw=0.8)
        # axes[0].plot(t, self.potential_energy(traj), label="Potentielle", lw=0.8)
        # axes[0].plot(t, self.spring_energy(traj), label="Ressorts", lw=0.8)
        # axes[0].plot(t, self.classical_kinetic_energy(traj)+self.spring_energy(traj)+self.potential_energy(traj), label="total", lw=0.8)
        axes[0].set_ylabel("Énergie (J)")
        axes[0].legend()
        axes[0].set_title("Énergies en fonction du temps")

        axes[1].plot(t, traj.X, lw=0.5, alpha=0.6)
        axes[1].set_ylabel("$x_s$ (m)")
        axes[1].set_title(f"Positions des {self.P} billes")

        axes[2].plot(t, self.centroid(traj), color="k", lw=1)
        axes[2].axhline(0, color="gray", ls="--", lw=0.5)
        axes[2].set_ylabel(r"Centroïde $\bar{x}$ (m)")
        axes[2].set_xlabel("Temps (s)")
        axes[2].set_title("Centre du polymère")

        fig.tight_layout()
        if filename:
            fig.savefig(filename, dpi=150)
        if show:
            plt.show()
        return fig


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------
def main(n_steps=10000, P=200, seed=None, show=True):
    a = 0.04e-9
    C = 0.8
    mass = c.m_p

    V0 = hbar**2 / (2 * mass * a**2 * C)
    pot = DoubleWellPotential(V0, a)
    # T = 0.2 * pot.V0 / kb
    T = 100
    print(f"V_0 = {pot.V0:.2e} J = {pot.V0 / c.e:.2f} eV")

    sim = RingPolymerLangevin(pot, mass, T, P=P,gamma=None, seed=seed,alpha = 1/100)
    print(f"dt = {sim.dt:.3e} s, P = {P}")

    traj = sim.run(n_steps)
    print(f"Fichier sauvegardé : {sim.save_csv(traj)}")
    sim.plot(traj, filename="trajectoires.png", show=show)
    return sim, traj


if __name__ == "__main__":
    main()