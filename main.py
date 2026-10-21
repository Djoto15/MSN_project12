import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
import csv

# --- Constantes du problème ---
a = 0.04e-9
C = 0.8
m = c.m_p
hbar = c.hbar
kb = c.k

P = 1   # P replicas to describe the proton

V_0 = hbar**2/(2*m*a**2*C)
T = 0.2*V_0/kb
w_p = P * kb * T / hbar   # pulsation d'oscillation des ressorts harmoniques
gamma = w_p # coefficient de friction: peut-être déterminé par calcul à partir de la corrèlation des forces





# --- Choix du pas de temps ---
alpha = 1/30    # on prend 30 pas dans la plus petite période d'oscillation (1/100 est idéal)
dt = alpha * 2*np.pi/w_p


# --- Configurqtion initiale ---
def vitesse_init():
    v_0 = np.sqrt(kb*P*T/m)  # écart-type des vitesses
    return np.random.normal(loc=0.0, scale=v_0, size=P)    # vitesse centré sur loc=0

def pos_init():
    return [0 for _ in range(P)]


# --- Fonctions utiles ---
def V_pot(x):
    return V_0*((x/a)**2-1)**2

def f(X,i,v,delta_t):
    '''Calculate the force on the i-th particle in the chain, given the position of the particles (X).'''
    P = len(X)
    dV = 4*V_0*(X[i]/a**2)*((X[i]/a)**2-1)

    omega_p = P*kb*T/hbar
    Ham_part = -dV - m*omega_p**2*(2*X[i]-X[(i-1)%P]-X[(i+1)%P])
    rdm_part = np.sqrt(2*m*gamma*kb*P*T/delta_t)*np.random.normal()
    relax_part = -gamma*m*v
    return Ham_part + rdm_part + relax_part

def v(x1,x2,x3,delta_t):
    '''calculate the velocity of a particle given its position at three consecutive time steps (x1, x2, x3) and the time step size (delta_t).'''
    return (3*x1 - 4*x2 + x3)/(2*delta_t)


def energy(X, V):
    '''Calculate the total energy of the system given the positions (X) and velocities (V) of the particles.'''
    P = len(X)
    omega_p = P*kb*T/hbar
    kinetic_energy = 1/2 * m * np.sum(V**2)- 1/2*m*omega_p**2*np.sum((X[1:]-X[:-1])**2) ##Pas sur de sum ou mean
    potential_energy = np.mean(V_pot(X))
    return kinetic_energy , potential_energy



# --- Algorithme de Verlet ---
def main(nb_iterations, delta_t, P):
    '''Main function to simulate the dynamics of a chain of particles in a double-well potential.'''
    X0 = pos_init()  # Initial positions
    V0 = vitesse_init()
    X = [X0]
    V = [V0]
    for t in range(nb_iterations):
        X.append(np.zeros(P))
        V.append(np.zeros(P))
        for i in range(P):
            force = f(X[t], i, V[t][i], delta_t)
            if t == 0:
                X[1][i] = X0[i] + V0[i] * delta_t + delta_t**2/(2*m)*force
                V[1][i] = V0[i] + force * delta_t / m
            # elif t==1:
            #     X[2][i] = 2*X[1][i] - X0[i] + delta_t**2/m*force
            #     V[2][i] = V[1][i] + force * delta_t / m
            else:
                X[t + 1][i] = 2*X[t][i] - X[t - 1][i] + delta_t**2/m*force
                V[t + 1][i] = v(X[t+1][i], X[t][i], X[t - 1][i], delta_t)
    return X, V


results = main(100, dt, P)

nom_fichier = "resultats.csv"
# 'newline=""' évite les lignes vides superflues sur Windows
with open(nom_fichier, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # Écriture de l'en-tête
    writer.writerow(["bead_index", "position_m", "vitesse_m_s"])
    
    # Écriture de l'ensemble des lignes
    writer.writerows(results)

print(f"Fichier sauvegardé : {nom_fichier}")


def plot_trajectories(X, V, delta_t):
    """
    X, V      : tableaux (n_steps, P) -> positions et vitesses de chaque bille à chaque pas
    potential : fonction V(x)
    delta_t   : pas de temps (pour l'axe des abscisses ; laisser 1 pour tracer en steps)
    """
    n_steps, P = len(X), len(X[0])
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


plot_trajectories(np.array(results[0]), np.array(results[1]), dt)