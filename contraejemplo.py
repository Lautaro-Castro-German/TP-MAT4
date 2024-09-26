import numpy as np
import matplotlib.pyplot as plt


# Definimos la función y su derivada
def f(x):
    return x**2 - 4


def grad_f(x):
    return 2 * x


# Parámetros del descenso por gradiente
x_init = 5.0  # Punto inicial
n_iterations = 1000  # Número de iteraciones
learning_rate_good = 0.1  # Tamaño de paso adecuado
learning_rate_bad = 1.5  # Tamaño de paso demasiado grande


# Función para ejecutar el descenso por gradiente
def gradient_descent(x_init, learning_rate, n_iterations):
    x_values = [x_init]
    for _ in range(n_iterations):
        x_new = x_values[-1] - learning_rate * grad_f(x_values[-1])
        x_values.append(x_new)
    return np.array(x_values)


# Ejecutamos el descenso por gradiente con un buen y mal tamaño de paso
x_vals_good = gradient_descent(x_init, learning_rate_good, n_iterations)
x_vals_bad = gradient_descent(x_init, learning_rate_bad, n_iterations)

# Valores de x para graficar la función
# x_plot = np.linspace(-6, 6, 400)
x_plot = np.linspace(-20, 20, 1000)  # Rango más amplio y más puntos

y_plot = f(x_plot)

# Creamos el gráfico
plt.figure(figsize=(10, 6))

# Graficamos la función original f(x) = x^2 - 4
plt.plot(x_plot, y_plot, label=r"$f(x) = x^2$", color="blue", linewidth=2)

# Graficamos las trayectorias del descenso por gradiente como líneas
plt.plot(
    x_vals_good,
    f(x_vals_good),
    color="green",
    marker="o",
    linestyle="-",
    label="Paso adecuado (η=0.1)",
    zorder=5,
)
plt.plot(
    x_vals_bad,
    f(x_vals_bad),
    color="red",
    marker="o",
    linestyle="-",
    label="Paso grande (η=1.5)",
    zorder=5,
)

# Marcamos el punto mínimo de la función
plt.axvline(x=0, color="gray", linestyle="--", label="Mínimo global")

# Configuramos las etiquetas y ajustamos los límites para que se vea la función correctamente
# plt.ylim([-6, 30])  # Limitar el eje y para ver bien la función y los puntos
# plt.xlim([-6, 6])  # Limitar el eje x para ver el rango relevante

# Ajustamos los límites del gráfico para ver mejor cómo se aleja con un paso grande
plt.ylim([-20, 400])  # Ampliamos el límite superior del eje y
plt.xlim([-20, 40])  # Ampliamos el límite del eje x

# Configuramos etiquetas y título
plt.title("Descenso por Gradiente con Diferentes Tamaños de Paso")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

# Mostramos el gráfico
plt.show()
