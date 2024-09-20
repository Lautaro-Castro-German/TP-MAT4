import pandas as pd
import numpy as np
from scipy import stats as st


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
            "T": T
        }


def significance_test(data):
    b1 = data['b1'] 
    sigma2 = data['sigma2']
    Sxx = data['Sxx']
    n = data['n']
    alfa = 0.05

    T_b1 = np.abs(b1 / np.sqrt(sigma2 / Sxx))
    t_b1 = st.t.ppf(1-alfa/2, n-2)
    if(T_b1 > t_b1):
        results = "x tiene cierta influencia sobre y"
    else:
        results = "x no tiene influencia sobre y"

    return results


file_path = "../players_21.csv"
df = pd.read_csv(file_path, na_values=["", " ", "NA", "N/A", "nan", "NaN"])

columns_to_keep = [
    "age",
    "height_cm",
    "weight_kg",
    "overall",
    "potential",
    "value_eur",
    "wage_eur",
    "skill_moves",
]
df_filtered = df[columns_to_keep]

r_values = {}
y_column = "value_eur"
y_values = df_filtered[y_column].values
for x_column in columns_to_keep:
    if x_column != y_column:
        x_values = df_filtered[x_column].values
        stats = calculate_stats(x_values, y_values)
        r_values[x_column] = stats["r"]

best_x_column = max(r_values, key=r_values.get)
best_r_value = r_values[best_x_column]

for key, value in sorted(r_values.items(), key=lambda item: item[1], reverse=True):
    print(f"Columna: {key} -> r: {value}")

print(f"La columna '{best_x_column}' produce el mayor valor de r: {best_r_value}")
# por lo tanto hacemos el analisis con esta columna

print(significance_test(stats))

stats = calculate_stats(df_filtered[best_x_column].values, y_values)
print("Coeficientes de correlación lineal r y de determinación r² son: ", stats["r"], stats["R2"])


# df_cleaned_rows.to_csv("players_21_filtered2.csv", index=False)
