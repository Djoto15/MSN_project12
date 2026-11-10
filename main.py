import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as c

from utils import DoubleWellPotential, RingPolymerLangevin


dossier = "figures"
os.makedirs(dossier, exist_ok=True)  # crée le dossier s'il n'existe pas

kb = c.k
hbar = c.hbar


# =============================================================================
# Outils communs
# =============================================================================
def run_simulation(
    n_steps: int = 50000,
    P: int = 32,
    gamma: float = None,
    alpha: float = 1 / 100,
    xc: float = None,
    C: float = 0.8,
    T_facteur: float = 0.2,
    a: float = 0.04e-9,
    seed: int = 42,
):
    """Initialise le potentiel, configure la simulation et exécute la trajectoire."""
    mass = c.m_p
    V0 = hbar**2 / (2.0 * mass * a**2 * C)
    pot = DoubleWellPotential(V0, a)
    T = T_facteur * pot.V0 / kb

    print(f"V0 = {pot.V0:.2e} J ({pot.V0 / c.e:.2f} eV) | T = {T:.2f} K")
    print(f"P = {P} | n_steps = {n_steps} | alpha = {alpha}")

    sim = RingPolymerLangevin(
        potential=pot,
        mass=mass,
        T=T,
        P=P,
        xc=xc,
        gamma=gamma,
        alpha=alpha,
        seed=seed,
    )
    print(f"dt = {sim.dt:.3e} s | gamma = {sim.gamma:.3e} s^-1")

    # Exécution de la dynamique
    res = sim.run(n_steps)
    print("Simulation terminée.")

    return sim, res


def plot_trajectoire(res,dx = 0.01, distributions = True):
    """Énergies, positions des billes et centroïde en fonction du temps."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    axes[0].plot(res.t, res.e_kin_class_kelvin, label="Cinétique classique", lw=0.8)
    axes[0].plot(res.t, res.e_pot_kelvin, label="Potentielle", lw=0.8)
    axes[0].plot(res.t, res.e_pot_kelvin + res.e_kin_class_kelvin, label="Totale", lw=0.8)
    axes[0].set_ylabel("Énergie (K)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Énergies en fonction du temps")

    if distributions:
        a=0.04e-9
        xmax = np.ceil(np.max(np.abs(res.X/a)) / dx) * dx
        edges = np.arange(-xmax, xmax + dx, dx)
        centers = 0.5 * (edges[:-1] + edges[1:])
        nb = len(centers)

        counts, _ = np.histogram((res.X/a).ravel(), bins=edges)
        density = counts / (counts.sum() * dx) 

        axes[1].bar(centers, density, width=dx, align="center")
        axes[1].set_xlabel("x/a")
        axes[1].set_ylabel("densité")
    else:
        axes[1].plot(res.t, res.X, lw=0.5, alpha=0.6)
        axes[1].set_ylabel("$x_s$ (m)")
        axes[1].set_title(f"Positions des {res.P} images")

    axes[2].plot(res.t, res.centroid, color="k", lw=1)
    axes[2].axhline(0, color="gray", ls="--", lw=0.5)
    axes[2].set_ylabel(r"Centroïde $\bar{x}$ (m)")
    axes[2].set_xlabel("Temps (s)")
    axes[2].set_title("Centre du polymère")

    fig.tight_layout()
    return fig


def _style_axe(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(True, which="major", color="#d3d3d3", linestyle="--", linewidth=0.7)
    ax.grid(True, which="minor", color="#f0f0f0", linestyle=":", linewidth=0.5)


_STYLE_COURBE = dict(
    marker="o",
    markersize=7,
    markerfacecolor="white",
    markeredgewidth=2,
    markeredgecolor="#1f77b4",
    linewidth=2,
    color="#1f77b4",
)


# =============================================================================
# Question 2 - Vérification du modèle (P = 1, gamma = 0)
# =============================================================================
def question_2(n_steps=1000, C=0.8, T_facteur=0.2):
    """Sans thermostat et avec une seule bille : l'énergie totale doit être conservée."""
    sim, res = run_simulation(
        n_steps=n_steps, P=1, gamma=0, alpha=1 / 100, xc=None, C=C, T_facteur=T_facteur, seed=None
    )
    fig = plot_trajectoire(res)
    fig.savefig(os.path.join(dossier, "Q2.png"), dpi=300, bbox_inches="tight")

    e_tot = res.e_pot_kelvin + res.e_kin_class_kelvin
    print(f"Dérive relative de l'énergie totale : {(e_tot.max() - e_tot.min()) / np.mean(e_tot):.2e}")
    plt.show()


# =============================================================================
# Question 3 - Vérification du thermostat (P = 1, dt et gamma optimaux)
# =============================================================================
def question_3(n_steps=100000, C=0.8, T_facteur=0.2):
    """On prend P = 1 avec les valeurs optimales du pas de temps et de gamma
    (choisies dans RingPolymerLangevin) et on vérifie l'équipartition."""
    sim, res = run_simulation(
        n_steps=n_steps, P=1, gamma=None, alpha=1 / 100, xc=None, C=C, T_facteur=T_facteur, seed=None
    )
    t_instantanee = res.e_kin_class_kelvin
    t_cumulee = np.cumsum(t_instantanee) / np.arange(1, len(t_instantanee) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res.t * 1e12, t_instantanee, color="tab:blue", lw=0.4, alpha=0.4,
            label=r"$T_{\mathrm{cin}}$ instantanée")
    ax.plot(res.t * 1e12, t_cumulee, color="tab:red", lw=1.8,
            label=r"Moyenne cumulée $\langle T_{\mathrm{cin}} \rangle$")
    ax.axhline(sim.T, color="black", ls="--", lw=1.2, label=f"Consigne $T = {sim.T:.2f}$ K")

    ax.set_xlabel("Temps (ps)")
    ax.set_ylabel("Température (K)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Vérification du thermostat de Langevin : équipartition de l'énergie cinétique")

    fig.tight_layout()
    fig.savefig(os.path.join(dossier, "Q3.png"), dpi=300, bbox_inches="tight")
    print(f"<T_cin> = {t_cumulee[-1]:.2f} K  (consigne T = {sim.T:.2f} K)")
    plt.show()


# =============================================================================
# Question 4 - Énergie potentielle en fonction de P
# =============================================================================
def question_4(
    list_P=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
    n_steps=200000,
    C=0.8,
    T_facteur=0.2,
    fig_name="Q4.png"
):
    """Évolution de <V>/(kB T) avec le nombre de billes P pour en déduire P optimal."""
    list_P = list(list_P)
    e_pot_mean = []

    for p in list_P:
        print(f"---------- Simulation pour P={p} ----------")
        sim, res = run_simulation(
            n_steps=n_steps, P=p, gamma=None, alpha=1 / 100, xc=None, C=C, T_facteur=T_facteur, seed=None
        )
        # On écarte le premier quart de la trajectoire pour moyenner à l'équilibre
        eq_start = len(res.e_pot) // 2
        e_pot_equil = res.e_pot[eq_start:]
        e_pot_mean.append(np.mean(e_pot_equil) / (kb * sim.T))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(list_P, e_pot_mean, label=r"$\langle V \rangle / (k_B T)$", **_STYLE_COURBE)

    # Échelle log base 2 pour P
    ax.set_xscale("log", base=2)
    ax.set_xticks(list_P)
    ax.set_xticklabels([str(p) for p in list_P])

    _style_axe(
        ax,
        r"Nombre de billes $P$ (Trotter number)",
        r"$\langle V \rangle / (k_B T)$",
        rf"Convergence de l'énergie potentielle en fonction de $P$ (C={C}, T={T_facteur}$V_0/k_B$)",
    )

    # Valeur asymptotique estimée (plateau à grand P)
    val_plateau = e_pot_mean[-1]
    ax.axhline(val_plateau, color="#d62728", linestyle=":", linewidth=1.2,
               label=f"Plateau ($P={list_P[-1]}$) : {val_plateau:.2f} $k_B T$")

    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(dossier, fig_name), dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# Question 5 - Contrainte sur le centroïde
# =============================================================================
def question_5(n_steps=100000, P=1024, xc=0.0, C=0.8, T_facteur=0.2, fig_name="Q5.png"):
    """On impose la position xc du centroïde : il ne doit pas bouger au cours de la dynamique."""
    sim, res = run_simulation(
        n_steps=n_steps, P=P, gamma=None, alpha=1 / 100, xc=xc, C=C, T_facteur=T_facteur, seed=None
    )
    fig = plot_trajectoire(res)
    fig.savefig(os.path.join(dossier, fig_name), dpi=300, bbox_inches="tight")

    print(f"Centroïde moyenne: {(np.mean(res.centroid) * 1e10):.2e} Angström")
    print(f"Centroïde écart-type: {(np.std(res.centroid) * 1e10):.2e} Angström")
    print(f"Force moyenne sur le centroïde f_c = {np.mean(res.fc):.3e} N")
    plt.show()


# =============================================================================
# Question 6 - Force f_c(x_c) et potentiel effectif
# =============================================================================
def question_6(
    n_xc=20,
    xc_max=0.05e-9,
    P=512,
    n_steps=100000,
    C=0.8,
    T_facteur=0.2,
    fig_name="Q6.png"
):
    """Calcul de f_c pour différentes valeurs de x_c puis intégration
    pour obtenir le potentiel effectif E_p(x_c) = -∫ f_c dx_c."""
    X_c = np.linspace(0, xc_max, n_xc)
    dX = X_c[1] - X_c[0]
    F_c = []
    E_p = [0]
    for x_c in X_c:
        sim, res = run_simulation(
            n_steps=n_steps, P=P, gamma=None, alpha=1 / 100, xc=x_c, C=C, T_facteur=T_facteur, seed=None
        )
        F_c.append(np.mean(res.fc))
        E_p.append(E_p[-1] - F_c[-1] * dX)
    E_p = np.array(E_p[: len(X_c)])

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), dpi=150)

    axes[0].plot(X_c, F_c, label=r"$f_c$", **_STYLE_COURBE)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    _style_axe(axes[0], r"Position centroïde $x_c$ (m)", r"$f_c$ (N)",
               rf"Valeur moyenne de $f_c$ à chaque position $x_c$ (C={C}, T={T_facteur}$V_0/k_B$)")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

    axes[1].plot(X_c, E_p, label=r"$E_p$", **_STYLE_COURBE)
    _style_axe(axes[1], r"Position centroïde $x_c$ (m)", r"$E_p$ (J)",
               r"Potentiel effectif $E_p$ à chaque position $x_c$")
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(dossier, fig_name), dpi=300, bbox_inches="tight")

    V0 = sim.pot.V0
    barriere = E_p[0] - np.min(E_p)
    print(f"Hauteur de barrière effective : {barriere:.3e} J = {barriere / V0:.3f} V0 "
          f"(x_c du minimum = {X_c[np.argmin(E_p)] * 1e10:.3f} Angström)")
    plt.show()

_STYLE_COURBE_BONUS = dict(
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=2,
            linewidth=2,
        )

def question_bonus(
    n_xc=20,
    xc_max=0.05e-9,
    P=512,
    n_steps=200000,
    fig_name="Qbonus.png"
):
    """Calcul de f_c pour différentes valeurs de x_c puis intégration
    pour obtenir le potentiel effectif E_p(x_c) = -∫ f_c dx_c."""
    

    fig, axes = plt.subplots(1, 1, figsize=(8, 5), dpi=150)

    
    for C, T_facteur in [(0.8, 0.2), (0.3, 0.2), (0.3, 0.4)]:
        X_c = np.linspace(0, xc_max, n_xc)
        dX = X_c[1] - X_c[0]
        F_c = []
        E_p = [0]
        for x_c in X_c:
            sim, res = run_simulation(
                n_steps=n_steps, P=P, gamma=None, alpha=1 / 100, xc=x_c, C=C, T_facteur=T_facteur, seed=None
            )
            F_c.append(np.mean(res.fc[len(res.fc) // 2:]))  # on prend la moyenne sur la seconde moitié de la trajectoire
            E_p.append(E_p[-1] - F_c[-1] * dX)
        E_p = np.array(E_p[: len(X_c)])

        V0 = sim.pot.V0
        a = sim.pot.a
        axes.plot(X_c/a, E_p/V0, label=r"$E_p$"+f" (C={C}, T factor={T_facteur})", **_STYLE_COURBE_BONUS)
        _style_axe(axes, r"Position centroïde $x_c/a$", r"$E_p/V_0$",
                    r"Potentiel effectif réduit $E_p/V_0$ à chaque position $x_c/a$")
        axes.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(dossier, fig_name), dpi=300, bbox_inches="tight")

    barriere = E_p[0] - np.min(E_p)
    print(f"Hauteur de barrière effective : {barriere:.3e} J = {barriere / V0:.3f} V0 "
          f"(x_c du minimum = {X_c[np.argmin(E_p)] * 1e10:.3f} Angström= {X_c[np.argmin(E_p)] / sim.pot.a:.3f} a)")
    plt.show()
# =============================================================================
# Question 7 - Reprise de 4 à 6 avec T = 0.2 V0/kB et C = 0.3
# =============================================================================
def question_7_P():
    question_4(C=0.3, T_facteur=0.2, fig_name="Q7_4.png")


def question_7_centroide():
    question_5(C=0.3, T_facteur=0.2, fig_name="Q7_5.png")


def question_7_potentiel():
    question_6(C=0.3, T_facteur=0.2, fig_name="Q7_6.png")


# =============================================================================
# Question 8 - Reprise de 4 à 6 avec T = 0.4 V0/kB et C = 0.3
# =============================================================================
def question_8_P():
    question_4(C=0.3, T_facteur=0.4, fig_name="Q8_4.png")


def question_8_centroide():
    question_5(C=0.3, T_facteur=0.4, fig_name="Q8_5.png")


def question_8_potentiel():
    question_6(C=0.3, T_facteur=0.4, fig_name="Q8_6.png")


# =============================================================================
# MENU
# =============================================================================
MENU = {
    "2": ("Vérification du modèle (P=1, gamma=0)", question_2),
    "3": ("Vérification du thermostat (P=1, dt et gamma optimaux)", question_3),
    "4": ("<V> en fonction de P (C=0.3, T=0.2 V0/kB)", question_4),
    "5": ("Centroïde contraint en x_c", question_5),
    "6": ("f_c(x_c) et potentiel effectif (C=0.8, T=0.2 V0/kB)", question_6),
    "7a": ("Q4 avec C=0.3, T=0.2 V0/kB : <V> en fonction de P", question_7_P),
    "7b": ("Q5 avec C=0.3, T=0.2 V0/kB : centroïde contraint", question_7_centroide),
    "7c": ("Q6 avec C=0.3, T=0.2 V0/kB : potentiel effectif", question_7_potentiel),
    "8a": ("Q4 avec C=0.3, T=0.4 V0/kB : <V> en fonction de P", question_8_P),
    "8b": ("Q5 avec C=0.3, T=0.4 V0/kB : centroïde contraint", question_8_centroide),
    "8c": ("Q6 avec C=0.3, T=0.4 V0/kB : potentiel effectif", question_8_potentiel),
    "bonus": ("Bonus : comparaison de E_p(x_c) pour différents C et T", question_bonus),
}


def menu():
    while True:
        print("\n" + "=" * 70)
        print(" Projet 12 - RPMD : potentiel en double puits")
        print("=" * 70)
        for cle, (description, _) in MENU.items():
            print(f"  [{cle:>2}]  Question {cle} - {description}")
        print("  [exit]  Quitter")
        print("-" * 70)

        choix = input("\nQuestion a afficher ? ").strip().lower()
        if choix in ("exit", ""):
            break
        if choix not in MENU:
            print(f"Choix '{choix}' invalide.")
            continue

        description, fonction = MENU[choix]
        print(f"\n>>> Question {choix} : {description}\n")
        fonction()


if __name__ == "__main__":
    menu()
