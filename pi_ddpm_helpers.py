from pi_ddpm_aux import *

def integrate_taylor_onestep(t, mu, dim, sigma, s, A, b, c):
    with torch.no_grad():
        t_tensor = torch.tensor([t]).reshape(1,1).float()
        #gamma = torch.zeros(dim).reshape(1,-1) #TODO: calculate gamma in a more sophisticated way
        gamma = torch.randn(dim).reshape(1,-1)
        mu_gamma_func = lambda x: mu(x, t_tensor)
        mu_gamma = mu_gamma_func(gamma).reshape(-1,1)
        J = autograd.functional.jacobian(mu_gamma_func, gamma)[0,:,0,:]
        A1_p_A2_inv = torch.inverse(A + 1/sigma*J.T @ J)

        #p(x_tau-1) via n-dim Guassian integral
        #s = s * (2*math.pi*sigma**2)**(-dim/2) * (torch.det(2*math.pi*A1_p_A2_inv))**(-1/2) #why does this screw things up?
        A = 1/sigma*torch.eye(dim) \
            - 1/sigma**2 * J @ A1_p_A2_inv @ J.T
        
        gamma = gamma.reshape(-1,1)
        b = b.reshape(-1,1)

        c = (1/2)*(b.T + 1/sigma*gamma.T@J.T@J - 1/sigma*mu_gamma.T@J) @ A1_p_A2_inv @ (b + 1/sigma*J.T@J@gamma - 1/sigma*J.T@mu_gamma) \
            + c - 1/(2*sigma)*(mu_gamma.T@mu_gamma + gamma.T@J.T@J@gamma) + 1/sigma*mu_gamma.T@J@gamma

        b = 1/(2*sigma) * (b.T + 1/sigma*gamma.T@J.T@J - 1/sigma*mu_gamma.T@J) @ (A1_p_A2_inv.T + A1_p_A2_inv) @ J.T \
            + 1/sigma * (mu_gamma.T - gamma.T@J.T)
        
    return s, A, b.T.view(-1), c.view(-1)

def make_t_tensor(t_pts, t_idxs, n_repeats=0): #absolute index of t (not batch index)
    if n_repeats:
        return torch.tensor(t_pts[t_idxs]).repeat(n_repeats).view(-1, 1).requires_grad_(True).float()
    return torch.tensor(t_pts[t_idxs]).view(-1, 1).requires_grad_(True).float()

def index_positional_encoding(seq_len, freqs, max_period=10000):
    assert freqs % 2 == 0, 'freqs must be even'
    positions = torch.arange(0, seq_len).unsqueeze(1).float() #(seq_len, 1)
    div_term = torch.exp(-math.log(max_period) * torch.arange(0, freqs, 2).float() / freqs) #[1, d_model//2]
    pos_enc = torch.zeros(seq_len, freqs)
    pos_enc[:, 0::2] = torch.sin(positions * div_term) #sine to even indices
    pos_enc[:, 1::2] = torch.cos(positions * div_term) #cosine to odd indices
    return pos_enc #(seq_len, freqs)

def value_positional_encoding(step_val, freqs, max_period=10000): #input = (bsz, seq_len, 1), max_period => t=100 @ dt=0.01
    assert freqs % 2 == 0, 'freqs must be even'
    half_dim = freqs//2
    freq = torch.exp(-math.log(max_period)* torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
    freq = freq.view(1, 1, half_dim)
    step_freq = step_val * freq
    sin_part = torch.sin(step_freq)
    cos_part = torch.cos(step_freq)
    pos_enc = torch.cat([sin_part, cos_part], dim=-1)
    return pos_enc #(bsz, seq_len, freqs)    

def eta_format(val):
    if val < 60: #only seconds
        return f'ETA: {val:.1f} s'
    elif val < 3600: #mins, secs
        return f'ETA: {val//60:.0f} m {round(val%60):.0f} s'
    else: #hrs, mins
        return f'ETA: {val//3600:.0f} h {round((val%3600)/60):.0f} m'

def unique_geometric_ints(T, num_points):
    candidate = np.geomspace(1, T, num=num_points, dtype=int)
    candidate[0] = 1
    candidate[-1] = T
    iteration = 0
    while len(np.unique(candidate)) != num_points:
        iteration += 1
        if iteration > 10000:
            raise ValueError("Failed to converge to a unique sequence within max iterations.")
        for i in range(1, num_points):
            if candidate[i] <= candidate[i - 1]:
                candidate[i] = candidate[i - 1] + 1
        if candidate[-1] > T:
            excess = candidate[-1] - T
            for i in range(num_points - 2, 0, -1):
                available = candidate[i] - (candidate[i - 1] + 1)
                reduction = min(available, excess)
                candidate[i] -= reduction
                excess -= reduction
                if excess == 0:
                    break
            candidate[-1] = T
    return candidate

def TENS_DEBUG(*args):
    print('--start debug--')
    print(f'Shapes of {len(args)} tensors:')
    for i, arg in enumerate(args):
        print(f'{i+1}. {type(arg)}, {arg.shape}')
    print('--end debug--')


def plot_pairwise_joint(data, dim, dt, t_range=None, x_range=None):
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2)

    t_idxs = np.arange(int(t_range[0] / dt), int(t_range[1] / dt)) if t_range is not None else np.arange(
        data.shape[2] - 1)
    x_t = data[:, dim, t_idxs]
    x_tp1 = data[:, dim, t_idxs + 1]
    if x_range is not None:
        mask = (x_t >= x_range[0]) & (x_t <= x_range[1])
        x_t = x_t[mask]
        x_tp1 = x_tp1[mask]
    x_t = x_t.flatten()
    x_tp1 = x_tp1.flatten()

    # histogram
    ax0 = fig.add_subplot(gs[0])
    ax0.hist2d(x_t, x_tp1, bins=100)
    ax0.set_title('Empirical joint')
    ax0.set_xlabel('x(t)')
    ax0.set_ylabel('x(t+dt)')

    # KDE
    xy_data = np.vstack([x_t, x_tp1])
    kde = sp.stats.gaussian_kde(xy_data)
    x_min, x_max = x_t.min(), x_t.max()
    y_min, y_max = x_tp1.min(), x_tp1.max()
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    ax1 = fig.add_subplot(gs[1], projection='3d')
    # ax1.contour(X, Y, Z, colors='black')
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.view_init(80, -90)

    plt.show()


def plot_triplet_joint(data, dim, t_range, xtm1_range, xt_range):
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2)

    t_idxs = np.arange(int(t_range[0] / dt), int(t_range[1] / dt)) if t_range is not None else t_points
    x_tm1 = data[:, dim, t_idxs - 1]
    x_t = data[:, dim, t_idxs]
    x_tp1 = data[:, dim, t_idxs + 1]

    mask = ((x_t >= xt_range[0]) & (x_t <= xt_range[1]) & (x_tm1 >= xtm1_range[0]) & (x_tm1 <= xtm1_range[1]))
    x_tp1 = x_tp1[mask]
    x_tp1 = x_tp1.flatten()

    # histogram
    ax0 = fig.add_subplot(gs[0])
    ax0.hist(x_tp1, bins=100)
    ax0.set_title('Empirical freq')

    # KDE
    kde = sp.stats.gaussian_kde(x_tp1)
    # x_min, x_max = x_t.min(), x_t.max()
    # y_min, y_max = x_tp1.min(), x_tp1.max()
    # x_vals = np.linspace(x_min, x_max, 100)
    # y_vals = np.linspace(y_min, y_max, 100)
    # X, Y = np.meshgrid(x_vals, y_vals)

    # positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(np.linspace(-10, 10))  # .reshape(X.shape)

    ax1 = fig.add_subplot(gs[1])  # , projection='3d')
    # ax1.contour(X, Y, Z, colors='black')
    ax1.plot(Z)
    # ax1.view_init(80, -90)

    plt.show()

def truncated_geometric(p, max_val, size):
    ks = np.arange(1, max_val + 1)
    pmf = p * (1 - p)**(ks - 1)  # shape = (max_val,)
    pmf /= pmf.sum()
    return np.random.choice(ks, size=size, p=pmf)