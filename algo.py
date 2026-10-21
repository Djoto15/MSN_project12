import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c

P = 100   # P replicas to describe the proton
m = c.m_p
k_B = c.k
h_bar = c.hbar
T = 300 # 300K: température ambiante

w_p = P * k_B * T / h_bar   # pulsation d'oscillation des ressorts harmoniques

# --- Choix du pas de temps ---
alpha = 1/30    # on prend 30 pas dans la plus petite période d'oscillation (1/100 est idéal)
dt = alpha * 2*np.pi/w_p

# --- Génération des vitesses ---
v_0 = np.sqrt(k_B*T/m)  # écart-type des vitesses
v = np.random.normal(loc=0.0, scale=v_0, size=P)    # vitesse centré sur loc=0

# --- Génération des positions ---
x = [0 for _ in range(P)]


# # --- Tracé et comparaison théorique ---
# plt.figure(figsize=(8, 5))

# # Histogramme normalisé des vitesses
# plt.hist(v, bins=60, density=True, alpha=0.6, color='steelblue', label=f'Échantillon ($P={P}$)')

# # Courbe théorique
# v_axis = np.linspace(-4 * v_0, 4 * v_0, 500)
# pdf_theorique = (1 / (np.sqrt(2 * np.pi) * v_0)) * np.exp(-v_axis**2 / (2 * v_0**2))
# plt.plot(v_axis, pdf_theorique, 'r-', lw=2, label='Gaussienne théorique')

# plt.title("Distribution des vitesses 1D")
# plt.xlabel("Vitesse $v$ (m/s)")
# plt.ylabel("Densité de probabilité")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()