import numpy as np
import scipy.constants as c

a = 0.04e-9
C = 0.8
m = c.m_p
hbar = c.hbar
kb = c.k

V_0 = C*hbar**2/(2*m*a**2)
T = 0.2*V_0/kb
gamma = 1 #! TO BE DEFINED

def V(x):
    return V_0*((x/a)**2-1)**2

def f(X,i,v,delta_t):
    '''Calculate the force on the i-th particle in the chain, given the position of the particles (X).'''
    P = len(X)
    dV = 4*V_0*(X[i]/a**2)*((X[i]/a)**2-1)

    omega_p = P*kb*T/hbar
    Ham_part = -dV - m*omega_p**2*(2*X[i]-X[(i-1)%P]-X[(i+1)%P])/(2*a**2)
    rdm_part = np.sqrt(2*m*gamma*kb*T/delta_t)*np.random.normal()
    relax_part = -gamma*m*v
    return Ham_part + rdm_part + relax_part

def v(x1,x2,x3,delta_t):
    '''calculate the velocity of a particle given its position at three consecutive time steps (x1, x2, x3) and the time step size (delta_t).'''
    return (3*x1 - 4*x2 + x3)/(2*delta_t)

def main(nb_iterations, delta_t, P):
    '''Main function to simulate the dynamics of a chain of P particles in a double-well potential.'''
    X0 = np.zeros(P)  # Initial positions
    V0 = np.random.normal(loc=0.0, scale=1.0, size=P) #!TO CHANGE
    X = [X0]
    V = [V0]
    for t in range(nb_iterations):
        X.append(np.zeros(P))
        V.append(np.zeros(P))
        for i in range(P):
            force = f(X[t], i, V[t][i], delta_t)
            if t == 0:
                X[1][i] = X0 + V0 * delta_t + delta_t**2/(2*m)*force
                V[1][i] = V0 + force * delta_t / m
            elif t==1:
                X[2][i] = 2*X[1][i] - X0 + delta_t**2/m*force
                V[2][i] = V[1][i] + force * delta_t / m
            else:
                X[t + 1][i] = 2*X[t][i] - X[t - 1][i] + delta_t**2/m*force
                V[t + 1][i] = v(X[t][i], X[t-1][i], X[t - 2][i], delta_t)
    return X, V

def energy(X, V):
    '''Calculate the total energy of the system given the positions (X) and velocities (V) of the particles.'''
    P = len(X)
    omega_p = P*kb*T/hbar
    kinetic_energy = 1/2 * m * np.sum(V**2)- 1/2*m*omega_p**2*np.sum((X[1:]-X[:-1])**2) ##Pas sur de sum ou mean
    potential_energy = np.mean(V(X))
    return kinetic_energy , potential_energy