import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

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
    X = np.concatenate([np.ones((df.shape[0], 1)), df[cols].values], axis=1)

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


    print("-" * 50)
    print("Descenso por gradiente")

def initialize_params(n_features):
    beta = np.random.randn(n_features) * 0.01
    return beta

def add_ones_column(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

def normalize_data_min_max(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def stochastic_gradient_descent(X, y, alfa=0.001, iterations=100, normalize=True):
    X = add_ones_column(X)
    if(normalize):
        X = normalize_data_min_max(X)

    beta = initialize_params(X.shape[1])
    n_samples = X.shape[0] # Número de muestras

    for i in range(iterations):
        for j in range(n_samples):
            # Seleccionar una muestra individual
            x_j = X[j]
            y_j = y[j]

            prediction = np.dot(x_j, beta)

            # Calcular los gradientes
            error = prediction - y_j
            gradient = error * x_j

            # Actualizar los pesos
            beta -= alfa * gradient
            


        # Opcional: imprimir el costo para seguimiento
        #cost = np.mean((np.dot(X, beta) - y) ** 2)
        #print cost variables used
        #print(f'Iteracion {i+1}, X: {x_j}, y: {y_j}, Prediccion: {prediction}, Error: {error}, Gradiente: {gradient}')
        #print(f'Iteracion {i+1}, Costo: {cost}')
    
    return beta

y = df["value_eur"].values
X = df[cols].values

beta = stochastic_gradient_descent(X, y, normalize=True)

print(f"Coeficientes: {beta}")
