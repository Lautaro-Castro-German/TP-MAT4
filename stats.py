import numpy as np
import json
import sys
import matplotlib.pyplot as plt

def render_points(x, y, m=1, b=0):
    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, color='blue', marker='o')

    x_values = np.linspace(min(x), max(x), 100)
    y_values = m * x_values + b
    plt.plot(x_values, y_values, color='red', label=f'y = {m}x + {b}', linewidth=2)

    plt.title('Gráfico de Dispersión con Línea Recta', fontsize=14)
    plt.xlabel('Valores de X', fontsize=12)
    plt.ylabel('Valores de Y', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()

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
    Syy = np.sum((y - y_mean)**2)

    b1 = Sxy / Sxx
    b0 = y_mean - b1 * x_mean

    regy = b0 + b1*x

    SCE = np.sum((y - regy)**2)
    SCR = np.sum((regy - y_mean)**2)
    STC = Syy

    R2 = 1 - (SCE / STC) # o también SCR / STC

    r = Sxy / np.sqrt(Sxx * Syy)
    sigma2 = SCE / (n - 2)

    T = b1 / np.sqrt(sigma2 / Sxx)

    return n, x_mean, y_mean, \
        Sxy, Sxx, Syy, \
        b0, b1, \
        SCE, SCR, \
        R2, r, \
        sigma2, T

if __name__ == '__main__':
    with open(sys.argv[1]) as json_data:
        dataset = json.load(json_data)

    x = dataset['x']
    y = dataset['y']

    n, x_mean, y_mean, \
    Sxy, Sxx, Syy, \
    b0, b1, \
    SCE, SCR, \
    R2, r, \
    sigma2, T = calculate_stats(x, y)

    print(
        f"$n = {n}$\n\n"
        f"$\\bar{{x}} = {x_mean}$\n\n"
        f"$\\bar{{y}} = {y_mean}$\n\n"
        f"$S_{{xy}} = {Sxy}$\n\n"
        f"$S_{{xx}} = {Sxx}$\n\n"
        f"$S_{{yy}} = {Syy}$\n\n"
        f"$\\hat{{\\beta_0}} = {b0}$\n\n"
        f"$\\hat{{\\beta_1}} = {b1}$\n\n"
        f"$\\hat{{y}} = {b1}x + {b0}$\n\n"
        f"$\\sigma^2 = {sigma2}$\n\n"
        f"$SCE = {SCE}$\n\n"
        f"$SCR = {SCR}$\n\n"
        f"$R^2 = {R2}$\n\n"
        f"$r = {r}$\n\n"
        f"T = \\frac{{{b1}}}{{\\sqrt{{{sigma2} / {Sxx}}}}} = {T}$\n\n"
    )

    if (len(sys.argv) > 2):
        print(f"evaluation {sys.argv[2]}={b0 + b1 * float(sys.argv[2])}")

    render_points(x, y, b1, b0)
