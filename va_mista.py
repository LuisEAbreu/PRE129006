import numpy as np
import matplotlib.pyplot as plt
import pre

# Definir parâmetros de simulação

N = 100_000 # Número de realizações

# Experimento probabilístico

U = np.random.uniform(0, 4, N)
X = np.empty_like(U) # ou X = np.empty(N, dtype=float)
idx1 = U <= 1
idx2 = U > 1
X[idx1] = np.random.uniform(0,2,idx1.sum())      # Uniform(0,2)
X[idx2] = np.random.binomial(1, 2/3, idx2.sum()) # Bernoulli(2/3)

# Cálculo

dx = 0.01
xs = np.arange(-.5, 2.5, dx)
pdf_X_teo = (1/8) * (0<=xs) * (xs<2) # Apenas a parte contínua
pdf_X_sim = pre.hist(X, xs)
cdf_X_teo = (
    (0)                             * (xs<0) +
    (1/4 + xs/8)        *   (0<=xs) * (xs<1) +
    (7/8 + (xs-1)/8)    *   (1<=xs) * (xs<2) +
    (1)                 *   (2<=xs)
)
cdf_X_sim = np.cumsum(pdf_X_sim) * dx
ev_X_teo = 3/4
ev_X_sim = np.mean(X)

# Saída

plt.subplot(2,1,1)
plt.bar(xs, pdf_X_sim, width=1*dx, color="y")
plt.plot(xs, pdf_X_teo, linewidth=3, color="b")
plt.annotate("", xytext=(0,0), xy=(0,1/4), arrowprops=pre.arrowprops)
plt.annotate("", xytext=(1,0), xy=(1,1/2), arrowprops=pre.arrowprops)
plt.xlim(-0.2,2.2)
plt.ylim(-0.1, 0.6)
plt.xlabel("$x")
plt.ylabel("$f_X(x)$")
plt.grid()
plt.subplot(2,1,2)
plt.plot(xs, cdf_X_sim, linewidth=4, color="y")
plt.plot(xs, cdf_X_teo, linewidth=4, color="b")
plt.ylim(-0.2, 2.2)
plt.ylim(-0.2, 1.2)
plt.xlabel("$x")
plt.ylabel("$F_X(x)$")
plt.grid()
print(f"Teorico: E[X]: {ev_X_teo:g}")
print(f"Simulado: E[X]: {ev_X_sim:g}")
plt.show()