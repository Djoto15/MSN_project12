import numpy as np
import scipy.constants as c
import csv
import matplotlib.pyplot as plt

# --- Constantes du problème ---
a = 0.04e-9
C = 0.8
m = c.m_p
hbar = c.hbar
kb = c.k

P = 1  # Nombre de répliques (Trotter number)

# 1. Correction de la formule de V_0 : C = hbar^2 / (2 * m * a^2 * V_0)
V_0 = hbar**2 / (2 * m * a**2 * C)
T = 0.2 * V_0 / kb
T_th = P * T  # Température effective du thermostat de Langevin

w_p = P * kb * T / hbar             # Pulsation des ressorts du polymère
w_0 = np.sqrt(8 * V_0 / (m * a**2)) # Pulsation propre au fond d'un puits

# 2. Coefficient de friction à l'échelle physique du système (ex: 0.5 * w_0)
gamma = 0.5 * w_0

# 3. Choix du pas de temps basé sur la fréquence la plus rapide
w_max = max(w_0, 2 * w_p)
alpha = 1 / 50
dt = alpha * (2 * np.pi / w_max)


# --- Configuration initiale ---
def vitesse_init(P):
    v_0 = np.sqrt(kb * T_th / m)  # Écart-type avec T_th = P * T
    return np.random.normal(loc=0.0, scale=v_0, size=P)

def pos_init(P):
    # Initialisation au fond d'un puits (x = a) ou en 0
    return np.zeros(P)


# --- Fonctions physiques ---
def V_pot(x):
    return V_0 * ((x / a)**2 - 1)**2

def f_totale(X, v, delta_t):
    """Calcule le vecteur des forces pour les P perles simultanément."""
    P = len(X)
    # Dérivée du potentiel externe divisée par P
    dV_dx = 4 * V_0 * (X / a**2) * ((X / a)**2 - 1)
    force_ext = -(1.0 / P) * dV_dx

    # Force des ressorts harmoniques (conditions périodiques : np.roll)
    X_plus = np.roll(X, -1)
    X_moins = np.roll(X, 1)
    force_ressort = -m * w_p**2 * (2 * X - X_moins - X_plus)

    # Thermostat de Langevin à T_th = P * T
    force_rdm = np.sqrt(2 * m * gamma * kb * T_th / delta_t) * np.random.normal(size=P)
    force_relax = -gamma * m * v

    return force_ext + force_ressort + force_rdm + force_relax

def vitesse_retrograde(x_curr, x_prev1, x_prev2, delta_t):
    """Différence finie décentrée à l'ordre 2 pour la vitesse à l'instant courant."""
    return (3 * x_curr - 4 * x_prev1 + x_prev2) / (2 * delta_t)


def energy(X, V):
    '''Calculate the total energy of the system given the positions (X) and velocities (V) of the particles.'''
    P = len(X)
    omega_p = P*kb*T/hbar
    kinetic_energy = 1/2 * m * np.sum(V**2)- 1/2*m*omega_p**2*np.sum((X[1:]-X[:-1])**2) ##Pas sur de sum ou mean
    potential_energy = np.mean(V_pot(X))
    return kinetic_energy , potential_energy


# --- Algorithme de Verlet ---
def main(nb_iterations, delta_t, P):
    X = np.zeros((nb_iterations + 1, P))
    V = np.zeros((nb_iterations + 1, P))

    X[0] = pos_init(P)
    V[0] = vitesse_init(P)

    for t in range(nb_iterations):
        force = f_totale(X[t], V[t], delta_t)
        
        if t == 0:
            X[1] = X[0] + V[0] * delta_t + (delta_t**2 / (2 * m)) * force
            V[1] = (X[1] - X[0]) / delta_t
        else:
            X[t + 1] = 2 * X[t] - X[t - 1] + (delta_t**2 / m) * force
            # Utilisation de X[t+1], X[t] et X[t-1] pour avoir la vitesse à t+1 !
            V[t + 1] = vitesse_retrograde(X[t + 1], X[t], X[t - 1], delta_t)

    return X, V


# --- Exécution et sauvegarde ---
nb_steps = 10000
X_res, V_res = main(nb_steps, dt, P)

nom_fichier = "resultats.csv"
with open(nom_fichier, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "temps_s", "x_centroid_nm", "v_centroid_m_s", "E_pot_reduite"])
    
    for step in range(nb_steps + 1):
        x_c = np.mean(X_res[step])
        v_c = np.mean(V_res[step])
        e_pot = np.mean(V_pot(X_res[step])) / V_0
        writer.writerow([step, step * dt, x_c * 1e9, v_c, e_pot])

print(f"Simulation terminée. Fichier sauvegardé : {nom_fichier}")


def plot_trajectories(X, V, delta_t):
    """
    X, V      : tableaux (n_steps, P) -> positions et vitesses de chaque bille à chaque pas
    potential : fonction V(x)
    delta_t   : pas de temps (pour l'axe des abscisses ; laisser 1 pour tracer en steps)
    """
    n_steps, P = X.shape
    t = np.arange(n_steps) * delta_t
    omega_p = P*kb*T/hbar

    # Énergies à chaque pas
    kinetic_energy , potential_energy = energy(X, V)
    centroid = X.mean(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # 1) Énergies
    axes[0].plot(t, kinetic_energy, label="Energy kinetic", lw=0.8)
    axes[0].plot(t, potential_energy, label="Potential energy $\\langle V \\rangle$", lw=0.8)
    axes[0].set_ylabel("Energy")
    axes[0].legend()
    axes[0].set_title("Energies as a function of time")

    # 2) Toutes les billes
    for s in range(P):
        axes[1].plot(t, X[:, s], lw=0.5, alpha=0.6)
    axes[1].set_ylabel("$x_s$")
    axes[1].set_title(f"Positions des {P} billes")

    # 3) Centroïde
    axes[2].plot(t, centroid, color="k", lw=1)
    axes[2].axhline(0, color="gray", ls="--", lw=0.5)
    axes[2].set_ylabel("Centroïde $\\bar{x}$")
    axes[2].set_xlabel("Step" if delta_t == 1.0 else "Temps")
    axes[2].set_title("Centre du polymère")

    plt.tight_layout()
    plt.show()




plot_trajectories(X_res, V_res, dt)