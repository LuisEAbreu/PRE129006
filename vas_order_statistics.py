import numpy as np
import matplotlib.pyplot as plt
import pre

# Definir parâmetros de simulação

N = 100_000 # Número de realizações

# Experimento probabilístico

U = np.random.uniform(low=0, high=1, size=(3,N))
X = np.sort(U, axis=0) #axis=0 classifica as colunas

# Cálculos

dx = 0.02
xs = np.arange(-0.2, 1.2, dx)
pdf_U_teo = np.empty((3, xs.size))
pdf_U_teo[0] = [1 if 0 <= u <= 1 else 0 for u in xs]
pdf_U_teo[1] = [1 if 0 <= u <= 1 else 0 for u in xs]
pdf_U_teo[2] = [1 if 0 <= u <= 1 else 0 for u in xs]
pdf_U_sim = [pre.hist(U[i],xs) for i in range(3)]

pdf_X_teo = np.empty((3, xs.size))
pdf_X_teo[0] = [3*(1-x)*(1-x) if 0 <= x <= 1 else 0 for x in xs]
pdf_X_teo[1] = [6*y*(1-y) if 0 <= y <= 1 else 0 for y in xs]
pdf_X_teo[2] = [3*z*z if 0 <= z <= 1 else 0 for z in xs]
pdf_X_sim = [pre.hist(X[i],xs) for i in range(3)]

ev_U_teo = 1/2 * np.array([1,1,1])
ev_U_sim = np.mean(U, axis=1)
ev_X_teo = 1/4 * np.array([1,2,3])
ev_X_sim = np.mean(X, axis=1)

cov_U_teo = 1/12 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
cov_U_sim = np.cov(U)
cov_X_teo = 1/80 * np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]])
cov_X_sim = np.cov(X)

# Saída

plt.figure()
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.bar(xs, pdf_U_sim[i], width=0.8*dx, color="y")
    plt.plot(xs, pdf_U_teo[i], linewidth=3, color="b")
    plt.xlabel("$u$")
    plt.ylabel(f"$f_{{U_{i+1}}}(u)$")
    plt.grid()
    plt.subplot(2, 3, i+4)
    plt.bar(xs, pdf_X_sim[i], width=0.8*dx, color="y")
    plt.plot(xs, pdf_X_teo[i], linewidth=3, color="b")
    plt.xlabel("$x$")
    plt.ylabel(f"$f_{{X_{i+1}}}(x)$")
    plt.grid()
plt.tight_layout()
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
print(f"Teo: E[U] = {ev_U_teo}")
print(f"Sim: E[U] = {ev_U_sim}")
print(f"Teo: E[X] = {ev_X_teo}")
print(f"Sim: E[X] = {ev_X_sim}")
print(f"Teo: cov[U] =\n{cov_U_teo}")
print(f"Sim: cov[U] =\n{cov_U_sim}")
print(f"Teo: cov[X] =\n{cov_X_teo}")
print(f"Sim: cov[X] =\n{cov_X_sim}")
plt.show()