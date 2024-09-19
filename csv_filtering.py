import pandas as pd
import numpy as np


def calculate_r(x, y):
    sxi = np.sum(x)
    syi = np.sum(y)
    n = len(x)

    sumprod = 0.0
    sumxpow2 = 0.0
    for i in range(n):
        sumprod += x[i] * y[i]
        sumxpow2 += np.pow(x[i], 2)

    sxy = sumprod - ((sxi * syi) / n)
    sxx = sumxpow2 - (np.pow(sxi, 2) / n)
    # b1 = sxy / sxx

    # xmean = np.mean(x)
    ymean = np.mean(y)
    # b0 = ymean - b1 * xmean

    syy = 0.0
    for i in range(n):
        syy += np.pow(y[i] - ymean, 2)

    # stc = syy
    # sce = syy - (np.pow(sxy, 2) / sxx)
    # stdpow2 = sce / (n - 2)
    # r2 = 1 - (sce / stc)
    r = sxy / np.sqrt(sxx * syy)

    return r


file_path = "players_21.csv"
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
    # "pace",
    # "shooting",
    # "passing",
    # "dribbling",
    # "defending",
    # "physic",
]
df_filtered = df[columns_to_keep]

df_cleaned_rows = df_filtered.dropna()

# print(f"Filas antes de limpiar: {len(df_filtered)}")
# print(f"Filas después de limpiar: {len(df_cleaned_rows)}")

r_values = {}
y_column = "value_eur"
y_values = df_cleaned_rows[y_column].values
for x_column in columns_to_keep:
    if x_column != y_column:
        x_values = df_cleaned_rows[x_column].values
        r = calculate_r(x_values, y_values)
        r_values[x_column] = r

best_x_column = max(r_values, key=r_values.get)
best_r_value = r_values[best_x_column]

print(f"La columna '{best_x_column}' produce el mayor valor de r: {best_r_value}")
for key, value in r_values.items():
    print(f"{key}: {value}")

# df_cleaned_rows.to_csv("players_21_filtered2.csv", index=False)
