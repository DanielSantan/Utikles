# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('bmh')
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['figure.figsize'] = '7, 5'

def make_figure_axes(x, y, fig_number=1, fig_size=8):
    '''
    Creates a set of 3 axes to plot 2D function + marginals
    '''
    # determine max size
    size_x = x.max() - x.min()
    size_y = y.max() - y.min()
    max_size = max(size_x, size_y)
    min_size = min(size_x, size_y)

    if size_x >= size_y:
        fig_size_x = fig_size
        fig_size_y = (0.12 * fig_size_x +
                      0.65 * fig_size_x * min_size / max_size +
                      0.02 * fig_size_x +
                      0.18 * fig_size_x +
                      0.03 * fig_size_x)
        rect_main = [0.12,
                     0.12 * fig_size_x / fig_size_y,
                     0.65,
                     0.65 * fig_size_x * min_size / max_size / fig_size_y]
        rect_x = [0.12, ((0.12 + 0.65 * min_size / max_size + 0.02) *
                         fig_size_x / fig_size_y),
                  0.65, 0.18 * fig_size_x / fig_size_y]
        rect_y = [0.79, 0.12 * fig_size_x / fig_size_y,
                  0.18, 0.65 * fig_size_x * min_size / max_size / fig_size_y]
    else:
        fig_size_y = fig_size
        fig_size_x = (0.12 * fig_size_y +
                      0.65 * fig_size_y * min_size / max_size +
                      0.02 * fig_size_y +
                      0.18 * fig_size_y +
                      0.03 * fig_size_y)
        rect_main = [0.12 * fig_size_y / fig_size_x,
                     0.12,
                     0.65 * fig_size_y * min_size / max_size / fig_size_x,
                     0.65]
        rect_x = [0.12 * fig_size_y / fig_size_x, 0.79,
                  0.65 * fig_size_y * min_size / max_size / fig_size_x, 0.18]
        rect_y = [((0.12 + 0.65 * min_size / max_size + 0.02) *
                   fig_size_y / fig_size_x), 0.12,
                  0.18 * fig_size_y / fig_size_x, 0.65]

    fig = plt.figure(fig_number, figsize=(fig_size_x, fig_size_y))
    fig.clf()

    ax_main = fig.add_axes(rect_main)
    ax_marginal_x = fig.add_axes(rect_x, xticklabels=[])
    ax_marginal_y = fig.add_axes(rect_y, yticklabels=[])

    return ax_main, ax_marginal_x, ax_marginal_y


def plot_distribution(x, y, z, cmap='viridis'):
    x_limits = (x.min(), x.max())
    y_limits = (y.min(), y.max())

    ax_main, ax_marginal_x, ax_marginal_y = make_figure_axes(x, y)
    ax_main.pcolormesh(x, y, z, cmap=cmap)

    marginal_x = np.sum(z, axis=1)
    ax_marginal_x.plot(x[:, 0], marginal_x)
    [l.set_rotation(-90) for l in ax_marginal_y.get_xticklabels()]

    marginal_y = np.sum(z, axis=0)
    ax_marginal_y.plot(marginal_y, y[0])

    ax_main.set_xlim(x_limits)
    ax_main.set_ylim(y_limits)

    ax_marginal_x.set_xlim(x_limits)
    ax_marginal_y.set_ylim(y_limits)
    return ax_main, ax_marginal_x, ax_marginal_y



"""
extraccion de datos
"""
data = np.genfromtxt('GLB.Ts+dSST.csv', delimiter=',', filling_values=-.18)
year = data[:, 0]
grado_pol = 2
JD = data[:, 13]
"""
modelos
"""
data = [year, JD]


def modelo_1(yr, a0, a1, a2):
    return a0+a1*yr+a2*yr**2


def modelo_2(yr, A0, yr0, tau):
    return A0 + np.exp((yr - yr0) / tau)


def prior(beta, params):
    """
    Probabilidad a priori.
    """
    beta0, beta1, beta2 = beta
    mu0, sigma0, mu1, sigma1, mu2, sigma2 = params
    S = -1. / 2 * ((beta0-mu0)**2 / sigma0**2 + (beta1-mu1)**2 / sigma1**2 +
                   (beta2-mu2)**2 / sigma2**2)
    P = np.exp(S) / (2 * np.pi * sigma0 * sigma1)
    return P

def fill_prior(beta0_grid, beta1_grid, beta2_grid, prior_params):
    """
    Llena la grilla con las probabilidades a priori.
    """
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                output[i, j, k] = prior([beta0_grid[i, j, k],
                                        beta1_grid[i, j, k],
                                        beta2_grid[i, j, k]],
                                        prior_params)
    return output


def likelihood_1(beta, data):
    beta0, beta1, beta2 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - modelo_1(x, beta0, beta1, beta2))**2)
    L = (2 * np.pi * 1.5**2)**(-N / 2.) * np.exp(-S / 2 / 1.5**2)
    return L


def likelihood_2(beta, data):
    beta0, beta1, beta2 = beta
    x, y = data
    try:
        N = len(x)
    except:
        N = 1
    S = np.sum((y - modelo_2(x, beta0, beta1, beta2))**2)
    L = (2 * np.pi * 1.5**2)**(-N / 2.) * np.exp(-S / 2 / 1.5**2)
    return L


def fill_likelihood_1(beta0_grid, beta1_grid, beta2_grid, data):
    """
    Llena la grilla de valores de la verosimilitud.
    """
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                output[i, j, k] = likelihood_1([beta0_grid[i, j, k],
                                                  beta1_grid[i, j, k],
                                                  beta2_grid[i, j, k]],
                                                  data)
    return output

def fill_likelihood_2(beta0_grid, beta1_grid, beta2_grid, data):
    """
    Llena la grilla de valores de la verosimilitud.
    """
    output = np.zeros(beta0_grid.shape)
    ni, nj, nk = beta0_grid.shape
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                output[i, j, k] = likelihood_2([beta0_grid[i, j, k],
                                                 beta1_grid[i, j, k],
                                                 beta2_grid[i, j, k]],
                                                 data)
    return output

def chi1(data, pars):
    x, y = data
    chi2 = np.sum((y - modelo_1(x, *pars))**2)
    return chi2


def chi2(data, pars):
    x, y = data
    chi2 = np.sum((y - modelo_2(x, *pars))**2)
    return chi2

"""
fiteos de los modelos
"""

coef_fit_1, cov_1 = curve_fit(modelo_1, year, JD)
coef_fit_2, cov_2 = curve_fit(modelo_2, year, JD, p0=(0, 2000, 1))
#err = np.std(np.polyval(coef_fit_1[::-1], year))
#err2 = np.std(modelo_2(year, *coef_fit_2))
"""
coeficientes de los modelos
"""

print('coef modelo1:',coef_fit_1)
print('coef modelo2:',coef_fit_2)
"""
verosimilitudes
"""
print("verosimilitud ajustada modelo 1:", likelihood_1(coef_fit_1, data))
print("verosimilitud ajustada modelo 2:", likelihood_2(coef_fit_2, data))
"""
chi^2
"""
print(chi1(data, coef_fit_1))
print(chi2(data, coef_fit_2))
"""
Grillado
"""

beta0_grid, beta1_grid, beta2_grid = np.mgrid[0:6e2:61j, -6e-1:0:61j,
                                              4e-5:11e-5:61j]
beta_prior_pars = [3e2, 100, -3e-1, 100, 8e-5, 100]
prior_grid_1 = fill_prior(beta0_grid, beta1_grid, beta2_grid,
                             beta_prior_pars)
likelihood_grid_1 = fill_likelihood_1(beta0_grid, beta1_grid, beta2_grid,
                                            [year, JD])
post_grid_1 = likelihood_grid_1 * prior_grid_1
"""
setup ploteo
"""
plt.figure(1)
plt.plot(year, JD, label='Data')
plt.plot(year, np.polyval(coef_fit_1[::-1], year), label='Modelo1')
plt.plot(year, modelo_2(year, *coef_fit_2), label='Modelo2')
plt.xlabel('Años')
plt.ylabel('Promedio de temperatura anual[°C]')
plt.legend()
plt.grid()
plt.draw()
plt.show()



"""
grilla modelo 1
"""
beta0_grid, beta1_grid, beta2_grid = np.mgrid[-17e-1:3e-1:61j,
                                      1600:2600:61j,
                                      1:20e1:61j]
n0, n1, n2 = beta0_grid.shape
prior_m1 = np.zeros((n0, n1, n2))
likelihood_m1 = np.zeros((n0, n1, n2))

for i in range(n0):
    for j in range(n1):
        for k in range(n2):
            prior_m1[i, j, k] = prior([beta0_grid[i, j, k],
                                        beta1_grid[i, j, k],
                                        beta2_grid[i, j, k]],
                                        [1, 20, 10, 20, -10, 20])
            likelihood_m1[i, j, k] = likelihood_1([beta0_grid[i, j, k],
                                                  beta1_grid[i, j, k],
                                                  beta2_grid[i, j, k]],
                                                  [year, JD])
"""
normalizacion modelo 1
"""
dx_1 = 6e3 / 100
dy_1 = 6 / 100
dz_1 = 7e-6 / 100
"""
integracion rapida por rectangulos
"""
P_E_1 = np.sum(post_grid_1) * dx_1 * dy_1 * dz_1
print('P_E_cuad', P_E_1)
"""
Probabilidades marginales de los parametros del modelo 1
"""
plt.figure(3)
plt.clf()
prob_beta0 = np.sum(post_grid_1, axis=(0, 2)) * dx_1 / P_E_1
plt.plot(beta0_grid[:, 0], prob_beta0)
plt.title('Prob. marginal de $a_0$')
plt.xlabel('Valores')
plt.ylabel('Probabilidad')

plt.figure(4)
plt.clf()
prob_beta1 = np.sum(post_grid_1, axis=(0, 2)) * dy_1 / P_E_1
plt.plot(beta1_grid[0], prob_beta1)
plt.title('Prob. marginal de $a_1$')
plt.xlabel('Valores')
plt.ylabel('Probabilidad')

plt.figure(5)
plt.clf()
prob_beta2 = np.sum(post_grid_1, axis=(0, 2)) * dz_1 / P_E_1
plt.plot(beta2_grid[0][0], prob_beta2)
plt.title('Prob. marginal de $a_2$' )
plt.xlabel('Valores')
plt.ylabel('Probabilidad')
"""
grilla modelo 2
"""

A0_grid, yr0_grid, tau_grid = np.mgrid[-17e-1:3e-1:61j,
                                      1600:2600:61j,
                                      1:20e1:61j]
beta_prior_pars_exp = [-3e-1, 100, 2e3, 100, 4e1, 100]
prior_grid_2 = fill_prior(A0_grid, yr0_grid,
                            tau_grid, beta_prior_pars_exp)

likelihood_grid_2 = fill_likelihood_2(A0_grid, yr0_grid,
                                          tau_grid, [year, JD])
post_grid_2 = likelihood_grid_2 * prior_grid_2
"""
Normalizacion modelo 2
"""
dx_2 = 0.5 / 60
dy_2 = 10e3 / 60
dz_2 = 150 / 60
P_E_2 = np.sum(post_grid_2) * dx_2 * dy_2 * dz_2
print('P_E_2', P_E_2)
"""
Probabilidades marginales de los parametros del modelo 2
"""
plt.figure(6)
plt.clf()
prob_A0 = np.sum(post_grid_2, axis=(0, 2)) * dx_2 / P_E_2
plt.plot(A0_grid[:, 0], prob_A0)
plt.title('Prob. marginal de $A_0$')
plt.xlabel('Valores')
plt.ylabel('Probabilidad')


plt.figure(7)
plt.clf()
prob_yr0 = np.sum(post_grid_2, axis=(0, 2)) * dy_2 / P_E_2
plt.plot(yr0_grid[0], prob_yr0)
plt.title('Prob. marginal de $yr_0$')
plt.xlabel('Valores')
plt.ylabel('Probabilidad')

plt.figure(8)
plt.clf()
prob_tau = np.sum(post_grid_2, axis=(0, 2)) * dz_2 / P_E_2
plt.plot(tau_grid[0], prob_tau)
plt.title('Prob. marginal de tau')
plt.xlabel('Valores')
plt.ylabel('Probabilidad')

"""
valor esperado de cada parametro
"""
max_beta0 = np.argmax(prob_beta0)
e_beta0 = beta0_grid[:, 0][max_beta0][0]
max_beta1 = np.argmax(prob_beta1)
e_beta1 = beta1_grid[0][max_beta1][0]
max_beta2 = np.argmax(prob_beta2)
e_beta2 = beta2_grid[0][0][max_beta2]
print('Esperanza a0:', e_beta0)
print('Esperanza a1:', e_beta1)                   
print('Esperanza a2:', e_beta2)                       

max_A0 = np.argmax(prob_A0)
max_yr0 = np.argmax(prob_yr0)
max_tau = np.argmax(prob_tau)
e_A0 = A0_grid[:, 0][max_A0][0]
e_yr0 = yr0_grid[0][max_yr0][0]
e_tau = tau_grid[0][0][max_tau]

print('Esperanza A0, : ', e_A0,)
print('Esperanza yr0: ', e_yr0)
print('Esperanza tau ',  e_tau)

"""
factor bayesiano
"""
print('P(D|M1) / P(D|M2) = ', P_E_1 / P_E_2)
