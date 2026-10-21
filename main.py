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

V_0 = C*hbar**2/(2*m*a**2)
T = 0.2*V_0/kb
gamma = 1 # coefficient de friction: peut-être déterminé par calcul à partir de la corrèlation des forces

w_p = P * kb * T / hbar   # pulsation d'oscillation des ressorts harmoniques



# --- Choix du pas de temps ---
alpha = 1/30    # on prend 30 pas dans la plus petite période d'oscillation (1/100 est idéal)
dt = alpha * 2*np.pi/w_p


# --- Configurqtion initiale ---
def vitesse_init():
    v_0 = np.sqrt(kb*T/m)  # écart-type des vitesses
    return np.random.normal(loc=0.0, scale=v_0, size=P)    # vitesse centré sur loc=0

def pos_init():
    return [0 for _ in range(P)]


# --- Fonctions utiles ---
def V(x):
    return V_0*((x/a)**2-1)**2

def f(X,i,v,delta_t):
    '''Calculate the force on the i-th particle in the chain, given the position of the particles (X).'''
    P = len(X)
    dV = 4*V_0*(X[i]/a**2)*((X[i]/a)**2-1)

    omega_p = P*kb*T/hbar
    Ham_part = -dV - m*omega_p**2*(2*X[i]-X[(i-1)%P]-X[(i+1)%P])
    rdm_part = np.sqrt(2*m*gamma*kb*T/delta_t)*np.random.normal()
    relax_part = -gamma*m*v
    return Ham_part + rdm_part + relax_part

def v(x1,x2,x3,delta_t):
    '''calculate the velocity of a particle given its position at three consecutive time steps (x1, x2, x3) and the time step size (delta_t).'''
    return (3*x1 - 4*x2 + x3)/(2*delta_t)



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
            elif t==1:
                X[2][i] = 2*X[1][i] - X0[i] + delta_t**2/m*force
                V[2][i] = V[1][i] + force * delta_t / m
            else:
                X[t + 1][i] = 2*X[t][i] - X[t - 1][i] + delta_t**2/m*force
                V[t + 1][i] = v(X[t][i], X[t-1][i], X[t - 2][i], delta_t)
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