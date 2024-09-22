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

    return X, y


if __name__ == "__main__":
    inicio = time.process_time()  # inicio conteo de tiempo de ejecución

    def ecm(b):
        y_est = X.dot(b)
        error = y_est - Y
        return (1 / (2 * n)) * np.sum(error**2)

    def ecm_grad(b):
        y_est = X.dot(b)
        error = y_est - Y
        return (1 / n) * X.T.dot(error)

    X, Y = load_data()
    n = X.shape[0]
    # función y derivada
    f = ecm
    df = ecm_grad
    tol = 1000  # tolerancia antes 1e-16
    step = 0.01
    iteration = 1

    # Descenso de gradiente
    beta = np.zeros(X.shape[1])  # valor inicial
    f_prev = f(beta)
    beta_new = beta - step * df(beta)
    f_new = f(beta_new)
    error = abs(f_new - f_prev)

    while error > tol:
        beta, f_prev = beta_new, f_new

        beta_new = beta - step * df(beta)
        f_new = f(beta_new)
        error = abs(f_new - f_prev)

        iteration += 1

    fin = time.process_time()

    for i in range(beta_new.size):
        print(f"B{i} = {beta_new[i]}")

    print(
        f"Tiempo de ejecución: {fin - inicio:.4f} segundos con {iteration} iteraciones"
    )
