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
