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


def add_row(data: dict, iteration, prev, new, fnew, error):
    data["iteration"].append(iteration)
    data["prev"].append(prev)
    data["new"].append(new)
    data["fnew"].append(fnew)
    data["error"].append(error)


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
    data = {"iteration": [], "prev": [], "new": [], "fnew": [], "error": []}
    # función y derivada
    f = ecm
    df = ecm_grad
    tol = 1e-6  # tolerancia antes 1e-16
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
        add_row(data, iteration, f_prev, f_new, f_new, error)

    fin = time.process_time()

    for i in range(beta_new.size):
        print(f"B{i} = {beta_new[i]}")

    print(
        f"Tiempo de ejecución: {fin - inicio:.4f} segundos con {iteration} iteraciones"
    )

    # Definir cuántas iteraciones finales deseas imprimir
    n_last_iterations = 5

    # Determinar el índice inicial para las últimas iteraciones
    start_index = max(0, len(data["iteration"]) - n_last_iterations)

    # Imprimir las últimas n iteraciones
    print(f"Últimas {n_last_iterations} iteraciones:")
    for i in range(start_index, len(data["iteration"])):
        print(
            f"Iteración {data['iteration'][i]}: "
            f"f(x_i) = {data['prev'][i]}, "
            f"f(x_i+1) = {data['fnew'][i]}, "
            f"error = {data['error'][i]}\n"
        )
"""         print(
            f"Iteración {data['iteration'][i]}\n"
            f"\\begin{{itemize}}\n"
            f"\\item f($x_i$) = {data['prev'][i]} $\\qquad$ f($x_{{i+1}}$) = {data['fnew'][i]}\n"
            f"\\item error = {data['error'][i]}\n"
            f"\\end{{itemize}}"
        ) """
