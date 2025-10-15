import numpy as np
import matplotlib.pyplot as plt
import pre

# Parâmetros

N = 100_000
mu = [0,0,0]
C = [[3, 3, 0],
     [3, 5, 0],
     [0, 0, 6]]

# Experimento

vet_x = np.random.multivariate_normal(mean=mu, cov=C, size=N)
X = vet_x[:,0]
Y = vet_x[:,1]
Z = vet_x[:,2]

W = X + 2*Y - Z + 5
Xcond_C = X[(0.9 < Y) & (Y <= 1.1)]
Xcond_D = X[(2.9 < Z) & (Z <= 3.1)]

# Cálculos

# a) Apenas f_{X,Y}

dx = 0.5; xs = np.arange(-9, 9, dx) # os valores de -9 a 9 foram calculados com regra do 4 sigma: 4*np.sqrt(5)
dy = 0.5; ys = np.arange(-9, 9, dy)
xx, yy = np.meshgrid(xs, ys, indexing="ij")

pdf_XY_teo = 1 / np.sqrt((2*np.pi)**2 * 6) * \
             np.exp(-(5*xx**2 - 6*xx*yy + 3*yy**2) / 12)

pdf_XY_sim = pre.hist2(X, Y, xs, ys)

# b)

dw = 0.5; ws = np.arange(-20, 30, dw)

pdf_W_teo = 1 / np.sqrt(2*np.pi * 41) * \
            np.exp(-(ws - 5)**2 / (2 * 41))
pdf_W_sim = pre.hist(W,ws)

# c)

pdf_Xcond_teo = 1 / np.sqrt(2*np.pi * 1.2) * \
                np.exp(-(xs - 0.6)**2 / (2 * 1.2))
pdf_Xcond_sim = pre.hist(Xcond_C, xs)

# d)

pr_teo = pre.phi(1 / np.sqrt(3)) - pre.phi(0)
pr_sim = np.mean((0 <= Xcond_D) & (Xcond_D <= 1))

# Saída

# a)
fig = plt.figure("a)")
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot_surface(xx, yy, pdf_XY_teo, cmap="coolwarm")
ax.set_xlabel("$x$")
ax.set_xlabel("$y$")
ax.set_xlabel("$f_{X,Y}(x,y)$")
ax.set_title("Teoria")
# ax.view_init(90, 270)
ax = fig.add_subplot(1, 2, 2, projection="3d")
ax.plot_surface(xx, yy, pdf_XY_sim, cmap="coolwarm")
ax.set_xlabel("$x$")
ax.set_xlabel("$y$")
ax.set_xlabel("$f_{X,Y}(x,y)$")
ax.set_title("Simulada")
# ax.view_init(90, 270)

# b)
plt.figure("b)")
plt.bar(ws, pdf_W_sim, width=0.8*dw, color="y")
plt.plot(ws, pdf_W_teo, linewidth=3, color="b")
plt.xlim(-9,9)
plt.xlabel("$b")
plt.ylabel("$f_X(x | Y=1)$")
plt.grid()

# c)
plt.figure("b)")
plt.bar(ws, pdf_W_sim, width=0.8*dw, color="y")
plt.plot(ws, pdf_W_teo, linewidth=3, color="b")
plt.xlim(-9,9)
plt.xlabel("$c")
plt.ylabel("$f_X(x | Y=1)$")
plt.grid()

# d)


plt.show()