from dataclasses import dataclass
from typing import Optional
import numpy as np
import scipy.constants as c

kb = c.k
hbar = c.hbar


class DoubleWellPotential:
    """V(x) = V0 ((x/a)^2 - 1)^2"""

    def __init__(self, V0: float, a: float):
        self.V0 = V0
        self.a = a

    def energy(self, x: np.ndarray) -> np.ndarray:
        return self.V0 * ((x / self.a) ** 2 - 1.0) ** 2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 4.0 * self.V0 * x / self.a**2 * ((x / self.a) ** 2 - 1.0)


@dataclass
class SimulationResults:
    """Conteneur stockant l'ensemble des grandeurs simulées et calculées."""

    # Paramètres d'exécution
    n_steps: int
    dt: float
    P: int
    T: float

    # Trajectoires brutes
    t: np.ndarray              # (n_steps + 1,) temps en secondes
    steps: np.ndarray          # (n_steps + 1,) indices entiers des pas
    X: np.ndarray              # (n_steps + 1, P) positions des billes (m)
    V: np.ndarray              # (n_steps + 1, P) vitesses des billes (m/s)

    # Observables structurales
    centroid: np.ndarray       # (n_steps + 1,) position du centroïde \bar{x} (m)
    fc: np.ndarray             # (n_steps + 1,) force moyenne sur le centroïde (N)

    # Énergies (en Joules)
    e_pot: np.ndarray          # (n_steps + 1,) énergie potentielle moyenne (1/P) \sum V(x_s)
    e_kin_class: np.ndarray    # (n_steps + 1,) énergie cinétique classique (1/P) \sum 1/2 m v_s^2
    e_spring: np.ndarray       # (n_steps + 1,) énergie des ressorts (1/P) \sum 1/2 m \omega_P^2 \Delta x^2
    e_kin_quant: np.ndarray    # (n_steps + 1,) énergie cinétique quantique (e_kin_class - e_spring)
    e_tot_class: np.ndarray    # (n_steps + 1,) énergie mécanique totale classique (e_pot + e_kin_class)

    # Énergies adimensionnées par (0.5 * k_B) (en Kelvin)
    @property
    def e_pot_kelvin(self) -> np.ndarray:
        return self.e_pot / (0.5 * kb)

    @property
    def e_kin_quant_kelvin(self) -> np.ndarray:
        return self.e_kin_quant / (0.5 * kb)

    @property
    def e_kin_class_kelvin(self) -> np.ndarray:
        return self.e_kin_class / (0.5 * kb * self.P)

    @property
    def e_spring_kelvin(self) -> np.ndarray:
        return self.e_spring / (0.5 * kb * self.P)


class RingPolymerLangevin:
    def __init__(
        self,
        potential: DoubleWellPotential,
        mass: float,
        T: float,
        P: int = 1,
        xc: Optional[float] = None,
        gamma: Optional[float] = None,
        alpha: float = 1 / 100,
        seed: Optional[int] = None,
    ):
        self.pot = potential
        self.m = mass
        self.T = T
        self.P = P
        self.xc = xc
        self.rng = np.random.default_rng(seed)

        # Pulsation de chaîne RPMD et pulsation harmonique
        self.w_p = P * kb * T / hbar
        self.w0 = np.sqrt(4.0 * potential.V0 / (mass * potential.a**2))

        # Choix de gamma et du pas de temps dt
        self.gamma = 0.1 * max(self.w_p, self.w0) / (2.0 * np.pi) if gamma is None else gamma
        self.dt = alpha * 2.0 * np.pi / max(self.w_p, self.w0)

        # Constantes pré-calculées
        self._noise_amp = (
            np.sqrt(2.0 * mass * self.gamma * kb * P * T / self.dt)
            if self.gamma > 0
            else 0.0
        )
        self._k_spring = mass * self.w_p**2

    # --- Configuration initiale ---
    def vitess_init(self) -> np.ndarray:
        v_0 = np.sqrt(kb * self.P * self.T / self.m)
        v = self.rng.normal(0.0, v_0, size=self.P)
        if self.xc is not None:
            v -= np.mean(v)  # Vitesse nulle pour le centroïde
        return v

    def pos_init(self) -> np.ndarray:
        if self.xc is None:
            return np.ones(self.P) * self.pot.a
        return np.ones(self.P) * self.xc

    # --- Forces ---
    def spring_force(self, x: np.ndarray) -> np.ndarray:
        if self.P == 1:
            return np.zeros_like(x)
        return -self._k_spring * (2.0 * x - np.roll(x, 1) - np.roll(x, -1))

    def force(self, x: np.ndarray, v: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Force totale agissant sur les P billes."""
        return (
            -self.pot.gradient(x)
            + self.spring_force(x)
            + self._noise_amp * noise
            - self.gamma * self.m * v
        )

    # --- Intégration Verlet ---
    def run(self, n_steps: int) -> SimulationResults:
        P, m, dt = self.P, self.m, self.dt
        X = np.empty((n_steps + 1, P))
        V = np.empty((n_steps + 1, P))
        fc = np.empty(n_steps + 1)

        X[0] = self.pos_init()
        V[0] = self.vitess_init()
        noise = self.rng.standard_normal((n_steps, P))

        # Pas t = 0 -> t = 1 (Taylor d'ordre 2)
        F0 = self.force(X[0], V[0], noise[0])
        fc[0] = np.mean(F0)
        if self.xc is not None:
            F0 -= np.mean(F0)

        X[1] = X[0] + V[0] * dt + (dt**2 / (2.0 * m)) * F0
        V[1] = V[0] + (F0 * dt) / m

        # Pas suivants
        dt2_m = dt**2 / m
        inv_2dt = 1.0 / (2.0 * dt)

        for t in range(1, n_steps):
            F_t = self.force(X[t], V[t], noise[t])
            fc[t] = np.mean(F_t)

            if self.xc is not None:
                F_t -= np.mean(F_t)

            X[t + 1] = 2.0 * X[t] - X[t - 1] + dt2_m * F_t
            # V[t + 1] = (X[t + 1] - X[t - 1]) * inv_2dt
            V[t + 1] = (3*X[t + 1] - 4*X[t] + X[t - 1]) * inv_2dt

        fc[-1] = np.mean(self.force(X[-1],V[-1], noise[-1]))

        # --- Calcul vectorisé de toutes les observables ---
        t = np.arange(n_steps + 1) * dt
        steps = np.arange(n_steps + 1)
        centroid = np.mean(X, axis=1)

        # Énergies (J)
        e_pot = np.mean(self.pot.energy(X), axis=1)
        e_kin_class = 0.5 * m * np.mean(V**2, axis=1)

        if P == 1:
            e_spring = np.zeros(n_steps + 1)
        else:
            dX = X - np.roll(X, -1, axis=1)
            e_spring = 0.5 * self._k_spring * np.mean(dX**2, axis=1)

        e_kin_quant = e_kin_class - e_spring
        e_tot_class = e_pot + e_kin_class

        return SimulationResults(
            n_steps=n_steps,
            dt=dt,
            P=P,
            T=self.T,
            t=t,
            steps=steps,
            X=X,
            V=V,
            centroid=centroid,
            fc=fc,
            e_pot=e_pot,
            e_kin_class=e_kin_class,
            e_spring=e_spring,
            e_kin_quant=e_kin_quant,
            e_tot_class=e_tot_class,
        )