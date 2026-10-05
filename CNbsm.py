import numpy as np
import matplotlib.pyplot as plt

"""
boundary conditions
we are sampling over x-space which is effectively S space
our heat equation sampling does not work at the boundaries - we need to impose a boundary value
therefore we say:
if S = 0, our option is worthless, therefore V(0,t) = 0
if S = infinity, V(inf, t) = S-Ee^(-r(T-t))

since V(S,t) = e^(alphax + betat)*u
u_0 = 0
u_M = exp((k+1)x_max/2 + (k+1)²τ/4) − exp((k-1)x_max/2 + (k-1)²τ/4)
we plug these into each update to ensure they are unchanged using g_0 and g_M

note we sample our x space into M chunks.
"""


def initial_condition(x_grid, k):
    term1 = np.exp((k+1)*x_grid/2)
    term2 = np.exp((k-1)*x_grid/2)
    return np.maximum(term1 - term2, 0)


def g_M(x_max, tau, r, vol):
    k = 2*r/(vol**2)
    term1 = np.exp((k+1)*x_max/2 + ((k+1)**2)*tau/4)
    term2 = np.exp((k-1)*x_max/2 + ((k-1)**2)*tau/4)
    return term1 - term2


def CN_step(u, dt, dx, g_0, g_M):
    a = dt/(dx**2)
    dimU = np.size(u)
    A = np.zeros([dimU, dimU])
    B = np.zeros([dimU, dimU])
    for i in range(1, dimU-1):
        A[i, i-1] = -a/2
        A[i, i] = 1+a
        A[i, i+1] = -a/2
        B[i, i-1] = a/2
        B[i, i] = 1-a
        B[i, i+1] = a/2
    A[0, 0] = 1
    A[dimU-1, dimU-1] = 1

    rhs = B@u
    rhs[0] = g_0
    rhs[-1] = g_M

    return np.linalg.solve(A, rhs)


def build_grids(vol, T, M, N):
    x_max = 4 * vol * np.sqrt(T)
    x_min = -x_max
    x = np.linspace(x_min, x_max, M)
    dx = x[1] - x[0]

    tau_max = (vol**2/2)*(T-t)
    tau_space = np.linspace(0, tau_max, N+1)
    dtau = tau_space[1] - tau_space[0]

    return x, x_max, dx, tau_space, tau_max, dtau


def solve_heat_equation(u, x_max, dx, tau_space, dtau, r, vol):
    g0 = 0
    N = tau_space.size - 1
    for n in range(N):
        tau_new = tau_space[n + 1]
        u = CN_step(u, dtau, dx, g0, g_M(x_max, tau_new, r, vol))
    return u


def transform_back(u, x, E, alpha, beta, tau_max):
    S_grid = E*np.exp(x)
    V = E*np.exp(alpha*x+beta*tau_max)*u
    return S_grid, V


def plot_price_curve(S_grid, V):
    plt.plot(S_grid, V)
    plt.xlabel('S')
    plt.ylabel('V (call price)')
    plt.title('Crank-Nicolson FD price vs S')
    plt.show()



#--- define call-price parameters
global t
E = 100
vol = 0.1
r = 0.05
T = 1
t = 0

#--- transformed parameters
tau = (vol**2/2)*(T-t)
alpha = -0.5*(2*r/(vol**2)- 1)
beta = -0.25*(2*r/(vol**2)+1)**2

#--- sample space
M = 100
N = 100
x, x_max, dx, tau_space, tau_max, dtau = build_grids(vol, T, M, N)
k = 2*r/(vol**2)

#--- initial u
u = initial_condition(x, k)

#--- solve
u = solve_heat_equation(u, x_max, dx, tau_space, dtau, r, vol)

#--- transform back
S_grid, V = transform_back(u, x, E, alpha, beta, tau_max)

print(S_grid)
print(V)
plot_price_curve(S_grid, V)

