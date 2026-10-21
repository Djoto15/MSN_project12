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

