import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_student


def render_points(x, y, m=1, b=0):
    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, color="blue", marker="o")

    x_values = np.linspace(min(x), max(x), 100)
    y_values = m * x_values + b
    plt.plot(x_values, y_values, color="red", label=f"y = {m}x + {b}", linewidth=2)

    plt.title("Gráfico de Dispersión con Línea Recta", fontsize=14)
    plt.xlabel("Valores de X", fontsize=12)
    plt.ylabel("Valores de Y", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()


def calcule_t(n, v, alpha):
    return t_student.ppf(1 - alpha, n - v)


def calcule_ic(n, xstar, alpha, b0, b1, x_mean, Sxx, sigma2):
    t = calcule_t(n, 2, alpha / 2)
    y_hat = b0 + b1 * xstar
    wide = t * np.sqrt(sigma2 * (1 / n + (xstar - x_mean) ** 2 / Sxx))
    ic = [y_hat - wide, y_hat + wide]
    print(f"y_hat = {y_hat}\n" f"wide = {wide}\n" f"t = {t}\n" f"sigma2 = {sigma2}\n")
    return ic


def calcule_ip(n, xstar, alpha, b0, b1, x_mean, Sxx, sigma2):
    t = calcule_t(n, 2, alpha / 2)
    y_hat = b0 + b1 * xstar
    wide = t * np.sqrt(sigma2 * (1 + 1 / n + (xstar - x_mean) ** 2 / Sxx))
    ip = [y_hat - wide, y_hat + wide]
    return ip


def calculate_stats(x, y):
    x = np.array(x)
    y = np.array(y)

    n = x.size
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sumxy = np.sum(x * y)
    sumx = np.sum(x)
    sumx2 = np.sum(x**2)
    sumy = np.sum(y)

    Sxy = sumxy - (sumx * sumy) / n
    Sxx = sumx2 - sumx**2 / n
    Syy = np.sum((y - y_mean) ** 2)

    b1 = Sxy / Sxx
    b0 = y_mean - b1 * x_mean

    regy = b0 + b1 * x

    SCE = np.sum((y - regy) ** 2)
    SCR = np.sum((regy - y_mean) ** 2)
    STC = Syy

    R2 = 1 - (SCE / STC)  # o también SCR / STC

    t = calcule_t(n, 2, 0.05 / 2)

    r = Sxy / np.sqrt(Sxx * Syy)
    sigma2 = SCE / (n - 2)

    T = b1 / np.sqrt(sigma2 / Sxx)

    return {
        "n": n,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "Sxy": Sxy,
        "Sxx": Sxx,
        "Syy": Syy,
        "b0": b0,
        "b1": b1,
        "SCE": SCE,
        "SCR": SCR,
        "R2": R2,
        "r": r,
        "sigma2": sigma2,
        "T": T,
        "t": t,
    }


def print_stats(stats):
    print(
        f"$n = {stats['n']}$\n\n"
        f"$\\bar{{x}} = {stats['x_mean']}$\n\n"
        f"$\\bar{{y}} = {stats['y_mean']}$\n\n"
        f"$S_{{xy}} = {stats['Sxy']}$\n\n"
        f"$S_{{xx}} = {stats['Sxx']}$\n\n"
        f"$S_{{yy}} = {stats['Syy']}$\n\n"
        f"$\\hat{{\\beta_0}} = {stats['b0']}$\n\n"
        f"$\\hat{{\\beta_1}} = {stats['b1']}$\n\n"
        f"$\\hat{{y}} = {stats['b1']}x + {stats['b0']}$\n\n"
        f"$\\sigma^2 = {stats['sigma2']}$\n\n"
        f"$SCE = {stats['SCE']}$\n\n"
        f"$SCR = {stats['SCR']}$\n\n"
        f"$R^2 = {stats['R2']}$\n\n"
        f"$r = {stats['r']}$\n\n"
        f"T = \\frac{{{stats['b1']}}}{{\\sqrt{{{stats['sigma2']} / {stats['Sxx']}}}}} = {stats['T']}$\n\n"
        f"$t = {stats['t']}$\n\n"
    )


if __name__ == "__main__":

    cols = [
        "age",
        "height_cm",
        "weight_kg",
        "overall",
        "potential",
        "wage_eur",
        "skill_moves",
    ]

    # Lectura de los datos del archivo CSV
    with open(sys.argv[1]) as csvfile:
        df = pd.read_csv(csvfile)
        y = df["value_eur"].values
        max_col = -1
        x_max = None
        for col in cols:
            x = df[col].values
            stats = calculate_stats(x, y)
            print(f"Columna: {col}\n" f"R^2 = {stats['R2']}\n" f"r = {stats['r']}\n")
            if stats["R2"] > max_col:
                max_col = stats["R2"]
                x_max = x

    print(f"{x_max[0]}")

    s = calculate_stats(x_max, y)

    render_points(x_max, y, s["b1"], s["b0"])

    print_stats(s)

    ip = calcule_ip(
        n=s["n"],
        xstar=s["x_mean"],
        alpha=0.05,
        b0=s["b0"],
        b1=s["b1"],
        x_mean=s["x_mean"],
        Sxx=s["Sxx"],
        sigma2=s["sigma2"],
    )

    ic = calcule_ic(
        n=s["n"],
        xstar=s["x_mean"],
        alpha=0.05,
        b0=s["b0"],
        b1=s["b1"],
        x_mean=s["x_mean"],
        Sxx=s["Sxx"],
        sigma2=s["sigma2"],
    )

    L_ic = ic[1] - ic[0]
    L_ip = ip[1] - ip[0]

    print(f"Intervalo de confianza: {ic}\n" f"Intervalo de predicción: {ip}\n")

    print(f"La proporción es: {L_ip / L_ic}\n")
