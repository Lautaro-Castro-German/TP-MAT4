import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_data():
    cols = [
        "age",
        "height_cm",
        "weight_kg",
        "overall",
        "potential",
        "wage_eur",
        "skill_moves",
    ]
    csvfile = open(sys.argv[1])
    df = pd.read_csv(csvfile)
    y = df["value_eur"].values
    X = df[cols].values

    # Normalización de las características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Agregar columna de unos para el término independiente
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    return X, y, scaler


def denormalize_coefficients(beta_norm, scaler):
    # Desnormalizar los coeficientes
    means = scaler.mean_  # medias originales
    stds = scaler.scale_  # desviaciones estándar originales

    # Desnormalizar los coeficientes para las variables (excepto el término independiente)
    beta_orig = np.zeros_like(beta_norm)
    beta_orig[1:] = beta_norm[1:] / stds

    # Desnormalizar el término independiente
    beta_orig[0] = beta_norm[0] - np.sum((beta_norm[1:] * means) / stds)

    return beta_orig


if __name__ == "__main__":
    inicio = time.process_time()  # inicio conteo de tiempo de ejecución

    X, Y, scaler = load_data()

    # función y derivada
    n = X.shape[0]
    f = lambda x: (1 / (2 * n)) * np.sum((X.dot(x) - Y) ** 2)
    df = lambda x: (1 / n) * X.T.dot(X.dot(x) - Y)
    x_prev = np.zeros(X.shape[1])  # valor inicial
    tol = 1e-16  # tolerancia
    step = 0.01  # Reduzco el learning rate a un valor más bajo
    iteration = 1
    data = {"iteration": [], "prev": [], "new": [], "fnew": [], "error": []}

    # Descenso de gradiente
    if df(x_prev).all() != 0:
        f_prev = f(x_prev)
        x_new = x_prev - step * df(x_prev)
        f_new = f(x_new)
        error = abs(f_new - f_prev)
        data["iteration"].append(iteration)

        while error > tol:
            iteration += 1
            x_prev, f_prev = x_new, f_new
            x_new = x_prev - step * df(x_prev)
            f_new = f(x_new)
            error = abs(f_new - f_prev)
            data["iteration"].append(iteration)

    # Desnormalizar los coeficientes
    beta_denormalized = denormalize_coefficients(x_new, scaler)

    print(f"Coeficientes normalizados: {x_new}")
    print(f"Coeficientes desnormalizados: {beta_denormalized}")

    for i in range(beta_denormalized.size):
        print(f"B{i} = {beta_denormalized[i]}")

    fin = time.process_time()  # finalizo conteo de tiempo de ejecución
    print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")
