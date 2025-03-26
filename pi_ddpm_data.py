import numpy as np

from pi_ddpm_aux import *

#General system
def generate_data(G, ic, n_samples, dim, t_points):
    t_0 = t_points[0]
    t_max = t_points[-1]
    t_count = len(t_points)
    data = np.empty((n_samples, dim, t_count))
    for n_idx, x_0 in enumerate(ic):
        try:
            sol = solve_ivp(G, (t_0, t_max), x_0, t_eval=t_points)
            if sol.y.shape[1] != t_count:
                raise ValueError(f'Error: sol.y.shape={sol.y.shape} at iter {n_idx}, x0={x_0}; (({dim}, {t_count}) expected)')
        except ValueError as e:
            print(e)
            continue
        data[n_idx,:,:] = sol.y #(dim, t_count)
    return data

#Lorenz63 system
def lorenz63_batch(xyzs, *, s, r, b):
    x = xyzs[:, 0]
    y = xyzs[:, 1]
    z = xyzs[:, 2]
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.stack([x_dot, y_dot, z_dot], axis=1)  #(N, 3)

def generate_lorenz63(N, t_steps, dt, ic, dt_out=0.01, s=10, r=28, b=8/3, f=0, forcing=None):
    if forcing:
        assert forcing in ['periodic', 'nonperiodic']
        assert f != 0
    dim = 3
    xyzs = np.empty((N, t_steps, dim))  #(N, t_steps, dim)
    xyzs[:, 0, :] = ic
    for i in range(1, t_steps):
        if forcing == 'periodic':
            t = i*dt
            rho = r + 3*math.sin(2*math.pi*f*t) #as per paper (periodic)
        elif forcing == 'nonperiodic':
            t = i*dt
            rho = r + 3*(math.sin(2*math.pi*f*t)/3 + math.sin(math.sqrt(3)*f*t)/3 + math.sin(math.sqrt(17)*f*t)/3) #nonperiodic
        else: rho = r
        xyzs[:, i, :] = xyzs[:, i - 1, :] + lorenz63_batch(xyzs[:, i - 1, :], s=s, r=rho, b=b) * dt
    return np.transpose(xyzs, (0, 2, 1))  # Shape: (N, dim, t_steps)
    #full_res = np.transpose(xyzs, (0, 2, 1))  # Shape: (N, dim, t_steps)
    #return full_res[:, :, ::int(dt_out/dt)]

#Lorenz96 system
def lorenz96_batch(x, t):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + 8

def generate_lorenz96(N, t_steps, dt, ic, dim):
    x = np.empty((N, t_steps, dim))
    t = np.arange(0, t_steps*dt, dt)
    for i in range(N):
        x0 = ic[i]
        x[i] = sp.integrate.odeint(lorenz96_batch, x0, t)
    return x.transpose(0, 2, 1)

#Baseline
def solve_char_pde_pointwise(a, f, U0, t_star, x_star, n_pts):
    #1. Integrate X(t,x0)/dt = a(X(t,x0),t) with X(0,x0) = x0
    t_eval = np.linspace(t_star, 0, n_pts)  # Time points from t_star to 0
    bwd_sol = solve_ivp(a, (t_star, 0), x_star, t_eval=t_eval)
    x0_star = bwd_sol.y[:, -1]

    #2. Integrate dz/dt = f(X(t,x0),t,z(t)) with z(0) = U0(x0)
    t_eval = np.linspace(0, t_star, 10000)  # Time points from t_star to 0
    fwd_sol = solve_ivp(f, (0, t_star), [U0(x0_star)], t_eval=t_eval)
    z_star = fwd_sol.y[:, -1]
    return z_star

def solve_char_pde_grid(G, Jx_G, U0, t_star, x_star_grid, y_star_grid):
    a = lambda t, x: np.array(G(t,x))
    f = lambda t, z: -z*np.trace(Jx_G(t,z)) #assumes pre-computed Jacobian; try np.gradient as well
    xy_size_x = x_star_grid.shape[0]
    xy_size_y = y_star_grid.shape[1]
    z_star_grid = np.empty((xy_size_x, xy_size_y))
    for i in range(xy_size_x):
        for j in range(xy_size_y):
            z_star_grid[i, j] = solve_char_pde_pointwise(a, f, U0, t_star, np.array([x_star_grid[i, j], y_star_grid[i, j]]), xy_size_x)
    return z_star_grid #(x,y,z)

