import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # Lectura de los datos
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
    scalerx = StandardScaler()
    X = scalerx.fit_transform(X)

    # Agregar columna de unos para el término independiente
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    # Cálculos para el modelo de regresión
    XTX = np.dot(X.T, X)
    XTX_inv = np.linalg.inv(XTX)
    XTy = np.dot(X.T, y)
    beta = np.dot(XTX_inv, XTy)
    y_pred = np.dot(X, beta)

    # Cálculos para el coeficiente de determinación y correlación
    n = df.shape[0]
    k = beta.shape[0]
    SSr = np.sum((y - y_pred) ** 2)
    STC = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - SSr / STC
    R2a = 1 - (1 - R2) * ((n - k) / (n - k - 1))
    r = np.sqrt(R2a)
    print(
        f"Resultados del modelo:\n"
        f"R^2 = {R2}\n"
        f"R^2 ajustado = {R2a}\n"
        f"r = {r}\n"
        f"Coeficientes:"
    )
    for i in range(len(beta)):
        print(f"B{i} = {beta[i]}")

    # Imprimir la ecuación de la regresión en formato y sombrero
    equation = "ŷ = {:.2f}".format(beta[0])  # Término independiente
    for i, col in enumerate(cols):
        equation += " + {:.2f} * {}".format(beta[i + 1], col)

    print("\nEcuación de la regresión lineal múltiple:")
    print(equation)

    # Librería externa para verificación de resultados
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    summary = model.summary()
    rsquared = model.rsquared
    rsquared_adj = model.rsquared_adj
    corr = np.corrcoef(y, model.fittedvalues)[0, 1]
    print("\nResultados de la librería statsmodels:")
    print(
        f"{summary}\n"
        f"R^2 = {rsquared}\n"
        f"R^2 ajustado = {rsquared_adj}\n"
        f"r = {corr}\n"
    )

    # Gráfico de valores reales vs predichos
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y_pred), max(y_pred)], color="red", linestyle="--")
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.title("Valores Reales vs Predichos")
    plt.grid(True)
    plt.show()

    # Gráfico de residuos
    residuos = y - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Valores Predichos")
    plt.ylabel("Residuos")
    plt.title("Gráfico de Residuos")
    plt.grid(True)
    plt.show()
