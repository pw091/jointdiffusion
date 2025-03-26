import torch
from pi_ddpm_aux import *
from pi_ddpm_helpers import *
from abc import ABC, abstractmethod
class EpsilonTheta(nn.Module):
    '''Denoising score matching model'''
    def __init__(self, sequence, tau):
        super(EpsilonTheta, self).__init__()
        self.seq = sequence #parent sequence object
        cond_dim = sequence.cond_dim
        data_dim = sequence.data_dim
        n_joint_cond = sequence.n_joint_cond
        n_joint_out = sequence.n_joint_out
        self.tau = tau

        self.beta = self.seq.beta(tau)
        self.alpha = self.seq.alpha(tau)
        self.alpha_bar = self.seq.alpha_bar(tau)
        self.sigma = self.seq.sigma(tau)

        dmodel = self.seq.dmodel
        tgt_pos_freqs = self.seq.posfreqs
        self.tgt_proj_layer = nn.Linear(data_dim, dmodel - tgt_pos_freqs)
        # tgt_pos_freqs = d_model #additive
        self.tgt_pos_enc = index_positional_encoding(n_joint_out, freqs=tgt_pos_freqs, max_period=n_joint_out)
        self.unproj_layer = nn.Linear(dmodel, data_dim)
        self.output_layer = nn.Linear(2 * n_joint_out * data_dim, n_joint_out * data_dim)

        #conditional: encoder-decoder
        if self.seq.n_joint_cond >= 1:
            self.transformer = nn.Transformer(d_model=dmodel, nhead=self.seq.nheads, num_encoder_layers=self.seq.nlayers, num_decoder_layers=self.seq.nlayers, dim_feedforward=self.seq.dff, dropout=0, batch_first=True)
            src_pos_freqs = self.seq.posfreqs  # 4 #v1=4, v2=16
            self.src_proj_layer = nn.Linear(cond_dim, dmodel - src_pos_freqs)
            # src_pos_freqs = d_model #additive
            self.src_pos_enc = index_positional_encoding(n_joint_cond, freqs=src_pos_freqs, max_period=n_joint_cond)

        #uncoditional: encoder only
        elif self.seq.n_joint_cond == 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=self.seq.nheads, dim_feedforward=self.seq.dff, dropout=0, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.seq.nlayers)

        else:
            raise ValueError(f'Invalid condition length: {self.seq.n_joint_cond}')

    @abstractmethod
    def forward(self):
        pass

    def _forward_std(self, x_cond, x_tau):
        bsz = x_tau.shape[0]
        skip = x_tau

        x_tau = x_tau.view(bsz, self.seq.n_joint_out, self.seq.data_dim)
        proj_tau = self.tgt_proj_layer(x_tau)  # (bsz, n_joint_out, d_model-tgt_pos_freqs)
        tgt_pos = self.tgt_pos_enc.unsqueeze(0).repeat(bsz, 1, 1)  # (bsz, n_joint_out, tgt_pos_freqs)
        tgt = torch.cat((proj_tau, tgt_pos), dim=2)  # (bsz, n_joint_out, d_model)
        # tgt = proj_tau + tgt_pos

        #unconditioned
        if self.seq.n_joint_cond == 0:
            x = self.transformer(tgt)

        #conditional
        else:
            x_cond = x_cond.view(bsz, self.seq.n_joint_cond, self.seq.cond_dim)
            proj_cond = self.src_proj_layer(x_cond) #(bsz, n_joint_cond, d_model-src_pos_freqs)
            src_pos = self.src_pos_enc.unsqueeze(0).repeat(bsz, 1, 1) #(bsz, n_joint_cond, src_pos_freqs)
            src = torch.cat((proj_cond, src_pos), dim=2) #(bsz, n_joint_cond, d_model)
            #src = proj_cond + src_pos
            x = self.transformer(src, tgt) #(bsz, n_joint_out, d_model)

        x = self.unproj_layer(x)  # (bsz, n_joint_out, data_dim)
        x = x.view(bsz, -1)  # (bsz, n_joint_out*data_dim)
        x = torch.cat((x, skip), dim=1)  # (bsz, 2*n_joint_out*data_dim)
        return self.output_layer(x)

    @abstractmethod
    def train(self):
        pass

    def _train_std(self, data, t_points, bsz_s, bsz_t, n_iters, optimizer, scheduler):
        n_samples = data.shape[0] #D = (n_samples, dim, t)
        dim = data.shape[1]
        n_t = data.shape[2] - self.seq.n_joint_cond * self.seq.cond_lag
        bsz = bsz_s * bsz_t
        n_batches_s = n_samples // bsz_s
        n_batches_t = n_t // bsz_t
        s_indices = np.array(range(n_samples))
        t_end_indices = np.array(range(len(t_points) - self.seq.cond_lag)) + self.seq.cond_lag #(1, 2, ..., max_t); -1 to account for t-1 -> t prediction
        assert bsz_s <= n_samples and n_samples % bsz_s == 0, f'uniform trajectory batches only: {n_samples}//{bsz_s}'
        assert bsz_t <= n_t and n_t % bsz_t == 0, f'uniform time batches only: {n_t}//{bsz_t}'
        assert dim == self.seq.data_dim, 'data/model dim mismatch'

        iter_losses = []  # training loss
        for itr in range(n_iters):
            batch_losses = []
            random.shuffle(s_indices)
            random.shuffle(t_end_indices)  # (1, 2, ..., max_t) -> (17, max_t, ..., 3)
            batch_pairs = list(it.product(range(n_batches_s), range(n_batches_t)))  # for i: for j: sample batch i, time batch j
            random.shuffle(batch_pairs)

            for s_batch_idx, t_batch_idx in batch_pairs:
                if (s_batch_idx, t_batch_idx) == batch_pairs[1]:
                    batch_start_time = time()

                s_abs_idxs = s_indices[s_batch_idx * bsz_s:(s_batch_idx + 1) * bsz_s]
                t_abs_ends = t_end_indices[t_batch_idx * bsz_t:t_batch_idx * bsz_t + bsz_t]
                def x0_func(t_idxs, num_t):
                    def x0_single_func(t_idxs):  # absolute index of t (not batch index)
                        #x0 = torch.tensor(data[s_abs_idxs, :, :][:, :, t_idxs]) #np forces this stupid sequential indexing
                        x0 = data[s_abs_idxs, :, :][:, :, t_idxs]
                        x0 = x0.permute(0, 2, 1)  # (bsz_s, bsz_t, dim); upon reshape: [x0(t0), x0(t1), ..., x1(t0), ...]
                        return x0.reshape(bsz, dim)  # (bsz, dim)
                    if num_t == 1:
                        return x0_single_func(t_idxs)
                    return torch.cat([x0_single_func(t_idxs - i) for i in reversed(range(num_t))],dim=1)  # (bsz, num_t*dim)

                if self.seq.out_mode == 'residual':
                    def xres_func(t_idxs, num_t):
                        assert self.seq.cond_lag == 1, 'residuals only support next-t prediction'
                        x_pre = x0_func(t_idxs - 1, self.seq.n_joint_cond) #[x(t-10), ..., x(t-1)]
                        x_post = x0_func(t_idxs, num_t) #[x(t-9), ..., x(t)]
                        return x_post - x_pre[:,-dim*num_t:] #x(t-9)-x(t-10), ..., x(t)-x(t-1)
                    loss = self.seq.loss_fn(self, bsz, dim, x0_func, xres_func, t_abs_ends, self.tau)
                elif self.seq.is_time_dependent:
                    def xt_func(t_idxs, num_t):
                        xt = torch.zeros((bsz, num_t, dim+1))
                        for i in reversed(range(num_t)):
                            x = x0_func(t_idxs-i, 1) #(bsz, dim)
                            t = make_t_tensor(t_points, t_abs_ends-i, bsz_s) #(bsz, 1)
                            xt[:, num_t-1-i, :] = torch.cat((x, t), dim=1) #(bsz, dim+1)
                        return xt.view(bsz, -1) #(bsz, num_t*(dim+1))
                    loss = self.seq.loss_fn(self, bsz, dim, xt_func, x0_func, t_abs_ends, self.tau)
                else:
                    loss = self.seq.loss_fn(self, bsz, dim, x0_func, x0_func, t_abs_ends, self.tau)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item()) #t-agnostic

                if itr == 1 and (s_batch_idx, t_batch_idx) == batch_pairs[1]:
                    tau_print_condn = self.tau==self.seq.tau_points[-1] #self.tau==self.seq.tau_points[0]
                    if tau_print_condn:
                        #print(f'Linear ({n_iters} iters) ' + eta_format((time() - batch_start_time) * n_iters * n_batches_s * n_batches_t * len(self.seq.tau_points)))
                        print('First tau ' + eta_format((time() - batch_start_time) * n_iters * n_batches_s * n_batches_t))
                        print('Total ' + eta_format((time() - batch_start_time) * np.sum(self.seq.tr_iters) * n_batches_s * n_batches_t))

            scheduler.step()
            iter_losses.append(np.mean(batch_losses))

        return iter_losses

class ModelSequence(ABC):
    '''Sequence of DDPM-like models'''
    def __init__(self, tau_points, data_dim, n_joint_cond, n_joint_out, loss, out_mode, cond_lag, is_time_dependent, dmodel, nheads, nlayers, dff, posfreqs, G=None, Jx_G=None, vram_saver=True):
        self.vram_saver = vram_saver

        self.T = max(tau_points) #descending order
        assert self.T == tau_points[0]
        assert cond_lag in (0, 1) #current step or next step only
        self.cond_lag = cond_lag #time index lag of output wrt conditioner (i.e. x_out(t) ~ cond(t-lag))

        assert loss in ('score', 'directrecon', 'noiserecon')
        assert not (loss=='recon' and len(tau_points)>1) #recon only supports tau={T}
        if loss=='score' and self.T != len(tau_points):
            warnings.warn(f'Number of taus ({len(tau_points)}) does not match T ({self.T})', UserWarning)

        assert out_mode in ('spatial', 'residual')
        self.out_mode = out_mode

        self.tau_points = tau_points
        self.data_dim = data_dim
        self.cond_dim = data_dim + is_time_dependent #flexibility; a time tensor can be your cond as well
        self.n_joint_cond = n_joint_cond
        self.n_joint_out = n_joint_out

        self.G = G #For PINN stuff
        self.Jx_G = Jx_G
        self.is_time_dependent = is_time_dependent

        self.dmodel = dmodel #NN
        self.nheads = nheads
        self.nlayers = nlayers
        self.dff = dff
        self.posfreqs = posfreqs

        self.loss = loss
        self.tr_iters = np.geomspace(500, 100, len(tau_points), dtype=int) #[100, 93, ..., 10]; more iters as tau:99->1
        self.lr_decays = np.geomspace(1e-3, 1e-5, len(tau_points)) #[1e-3, ..., 1e-4]; higher LR as tau:99->1
        self.exp_decays = np.geomspace(0.99, 0.9, len(tau_points)) #[0.999, ..., 0.9] bigger gamma => less LR decay as tau:99->1
        self.pert_rates = np.geomspace(1e-3, 1e-5, len(tau_points)) #[1e-3, ..., 1e-4] = smaller perturbation as tau:99->1

        beta_min = 0.1 #assuming 1 -> 100; 0.0001 #fast DDPM (assuming 1 -> 1000, not good)
        beta_max = 0.2 #0.02
        self.beta = lambda tau: beta_min + (beta_max-beta_min) * (tau - 1) / (tau_points[0] - 1) if len(tau_points) > 1 else (beta_min+beta_max)/2
        self.alpha = lambda tau: 1 - self.beta(tau)
        self.alpha_bar = lambda tau: math.prod(self.alpha(tau_val) for tau_val in range(1, tau+1))
        self.sigma = lambda tau: ((1-self.alpha_bar(tau-1)) / (1-self.alpha_bar(tau)) * self.beta(tau))**(1/2)

        self.model_dict = {} #each tau step's model
        for tau in tau_points:
            self.model_dict[tau] = None

    def loss_fn(self, nn, bsz, dim, cond_func, out_func, t_abs_ends, tau):
        args = {'nn':nn, 'bsz':bsz, 'dim':dim, 'cond_func':cond_func, 'out_func':out_func, 't_abs_ends':t_abs_ends, 'tau':tau}
        if self.loss == 'score':
            return self._score_loss(**args)
        elif self.loss == 'directrecon':
            assert cond_func == out_func, 'direct recon loss conditions on x only'
            return self._direct_recon_loss(**args)
        elif self.loss == 'noiserecon':
            return self._noise_recon_loss(**args)
        else:
            raise ValueError(f'Invalid loss function: {self.loss}')

    def _score_loss(self, nn, bsz, dim, cond_func, out_func, t_abs_ends, tau):
        if self.n_joint_cond >= 1:
            cond = cond_func(t_abs_ends-1, self.n_joint_cond).float() #x_cond or t_cond
        else:
            cond = None
        x_out = out_func(t_abs_ends, self.n_joint_out)
        eps = torch.randn(size=(bsz, self.n_joint_out * dim))
        x_tau = (self.alpha_bar(tau) ** 0.5 * x_out + (1 - self.alpha_bar(tau)) ** 0.5 * eps).float()
        nn_output = nn.forward(cond, x_tau) #predicts epsilon
        return torch.mean((eps - nn_output) ** 2)

    def _direct_recon_loss(self, nn, bsz, dim, cond_func, out_func, t_abs_ends, tau):
        x_cond = cond_func(t_abs_ends-1, self.n_joint_cond).float() #x_cond only
        x_out = out_func(t_abs_ends, self.n_joint_out)
        eps = torch.randn(size=(bsz, self.n_joint_out * dim))
        x_tau = (self.alpha_bar(tau) ** 0.5 * x_out + (1 - self.alpha_bar(tau)) ** 0.5 * eps).float()
        nn_output = nn.forward(x_cond, x_tau) #predicts xout - xcond
        return torch.mean((x_out - (nn_output + x_cond[:, -dim * self.n_joint_out:])) ** 2) #truncate {x_cond}_(t-n)^(t-1) to x_cond(t-1)

    def _noise_recon_loss(self, nn, bsz, dim, cond_func, out_func, t_abs_ends, tau):
        cond = cond_func(t_abs_ends-1, self.n_joint_cond).float() #x_cond or t_cond
        x_out = out_func(t_abs_ends, self.n_joint_out)
        eps = torch.randn(size=(bsz, self.n_joint_out * dim))
        x_tau = (self.alpha_bar(tau) ** 0.5 * x_out + (1 - self.alpha_bar(tau)) ** 0.5 * eps).float()
        nn_output = nn.forward(cond, x_tau) #predicts xout - xtau(xout)|xcond
        return torch.mean((x_out - (nn_output + x_tau)) ** 2) #truncate {x_cond}_(t-n)^(t-1) to x_cond(t-1)

    def train(self, data, t_points, bsz_s, bsz_t, save, checkpoint=False):
        #tensor first
        data = torch.tensor(data).float()

        plot_taus = [self.tau_points[0], self.tau_points[len(self.tau_points)//2], 1]
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3)

        pert_scale = 0
        #for tau in self.tau_points: #100 -> 1
        for tau in reversed(self.tau_points): #1 -> 100
            #tau_prev = tau + 1 #100 -> 1
            tau_prev = tau - 1 #1 -> 100
            #if checkpoint and tau >= checkpoint: #100 -> 1
            if checkpoint and tau <= checkpoint: #1 -> 100
                continue
            model = self.model_dict[tau].to('cuda') #VRAM management (if CPU default)
            #if tau != self.tau_points[0]: #transfer learning
            if tau != self.tau_points[-1]: #transfer learning
                prev = self.model_dict[tau_prev].state_dict()
                pert_scale = self.pert_rates[tau-1]
                for k in prev.keys():
                    prev[k] = prev[k] + pert_scale*torch.randn_like(prev[k])
                model.load_state_dict(prev)
                if save:
                    self._save_submodel(tau_prev, save) #if checkpoint, it'll re-saves the last loaded model (not a big deal)
                #self.model_dict[tau_prev].to('cpu') #VRAM management (if CUDA default)

            n_iters = 50 if tau == self.T else self.tr_iters[tau-1]
            lr = 1e-5 if tau == self.T else self.lr_decays[tau-1]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            gamma = 0.9 if tau == self.T else self.exp_decays[tau-1]
            scheduler = ExponentialLR(optimizer, gamma=gamma)

            iter_losses = model.train(data, t_points, bsz_s, bsz_t, n_iters, optimizer, scheduler)
            model.to('cpu') #VRAM management (if CPU default)

            if tau % 10 == 0 or tau < 5 or tau > 95:
                minifig = plt.figure(figsize=(20, 5))
                minigs = minifig.add_gridspec(1, 4)
                ax0 = minifig.add_subplot(minigs[0])
                ax0.plot(iter_losses)
                ax0.set_title(f'tau={tau}: n_iters={n_iters}, lr={lr:.2e}, gamma={gamma:.2f}, pert={pert_scale:.2e}')
                ax1 = minifig.add_subplot(minigs[1])
                ax1.plot(iter_losses)
                ax1.set_ylim(0, 0.01)
                ax2 = minifig.add_subplot(minigs[2])
                ax2.plot(iter_losses)
                ax2.set_ylim(0, 0.001)
                ax3 = minifig.add_subplot(minigs[3])
                ax3.plot(iter_losses)
                ax3.set_ylim(0, 0.0001)
                plt.show()

            if tau in plot_taus:
                print(f'tau = {tau} complete')
                ax = fig.add_subplot(gs[plot_taus.index(tau)])
                ax.set_title(f'tau = {tau}')
                ax.plot(iter_losses, label='Total Loss')
                if tau == plot_taus[0]:
                    ax.set_xlabel('Iterations')
                    ax.set_ylabel('Loss across all t')
                    ax.legend()

        if save:
            self._save_submodel(tau, save)
        print('Training complete')
        #model.to('cpu') #VRAM management (if CUDA default)
        plt.show()
        return

    def infer(self, x_cond, n, t=None, dt=None):
        if self.is_time_dependent:
            return self._infer_score_tdep(x_cond, n, t, dt)
        elif self.loss == 'score':
            return self._infer_score(x_cond, n)
        elif self.loss == 'directrecon':
            return self._infer_direct_recon(x_cond, n)
        elif self.loss == 'noiserecon':
            return self._infer_noise_recon(x_cond, n)
        else:
            raise ValueError(f'Invalid loss function: {self.loss}')

    def _infer_score_tdep(self, x_cond, n, t, dt):
        with torch.no_grad():
            t_tens = torch.tensor([t-i*dt for i in reversed(range(self.n_joint_cond))]).unsqueeze(1)
            x_cond = x_cond.view(self.n_joint_cond, self.data_dim)
            cond = torch.cat((x_cond, t_tens), dim=1).unsqueeze(0).repeat(n, 1, 1)
            x_tau = torch.randn((n, self.n_joint_out * self.data_dim))
            for tau in self.tau_points:  #x_T -> T -> x_{T-1} -> ... -> tau_tgt+1 -> x_{tau_tgt}
                z = torch.randn_like(x_tau) if tau > 1 else torch.zeros_like(x_tau)
                model = self.model_dict[tau].to('cuda') #VRAM management for CPU default
                fwd = model.forward(cond, x_tau) #epsilon_theta
                model.to('cpu') #VRAM management for CPU default
                #fwd = self.model_dict[tau].forward(cond, x_tau) #epsilon_theta
                x_tau = self.alpha(tau) ** (-1 / 2) \
                        * (x_tau - (1 - self.alpha(tau)) / (1 - self.alpha_bar(tau)) ** (1 / 2) * fwd) \
                        + self.sigma(tau) * z #x_{tau-1}(t)
        return x_tau + x_cond[-self.data_dim*self.n_joint_out:] if self.out_mode == 'residual' else x_tau #x_{tau_tgt}(t)

    def _infer_score(self, x_cond, n): #use "infer"
        with torch.no_grad():
            x_tau = torch.randn((n, self.n_joint_out * self.data_dim))
            for tau in self.tau_points:  #x_T -> T -> x_{T-1} -> ... -> tau_tgt+1 -> x_{tau_tgt}
                z = torch.randn_like(x_tau) if tau > 1 else torch.zeros_like(x_tau)
                model = self.model_dict[tau]
                #if self.vram_saver:
                #    model.to('cuda') #Doens't hurt if already on GPU
                fwd = model.forward(x_cond.repeat(n, 1).float(), x_tau)  #epsilon_theta
                #if self.vram_saver:
                #    model.to('cpu') #VRAM management for CPU default
                #fwd = self.model_dict[tau].forward(x_cond.repeat(n, 1).float(), x_tau) #epsilon_theta
                x_tau = self.alpha(tau) ** (-1 / 2) \
                        * (x_tau - (1 - self.alpha(tau)) / (1 - self.alpha_bar(tau)) ** (1 / 2) * fwd) \
                        + self.sigma(tau) * z #x_{tau-1}(t)
        return x_tau + x_cond[-self.data_dim*self.n_joint_out:] if self.out_mode == 'residual' else x_tau #x_{tau_tgt}(t)

    def _infer_direct_recon(self, x_cond, n): #use "infer"
        x_T = torch.randn((n, self.n_joint_out * self.data_dim))
        fwd = self.model_dict[self.T].forward(x_cond.repeat(n, 1).float(), x_T)
        return fwd + x_cond[-self.data_dim*self.n_joint_out:]

    def _infer_noise_recon(self, x_cond, n): #use "infer"
        x_T = torch.randn((n, self.n_joint_out * self.data_dim))
        fwd = self.model_dict[self.T].forward(x_cond.repeat(n, 1).float(), x_T)
        return fwd + x_T

    def make(self, train, save, path, checkpoint=False, **tr_args):
        if checkpoint:
            self._load_models(path, checkpoint) #checkpoint is last tau already trained (inclusive)
        if train:
            if save:
                self.train(**tr_args, save=path, checkpoint=checkpoint)
            else:
                warnings.warn('Training without saving')
                self.train(**tr_args, save=False)
            #self.train(**tr_args)
            #if save:
            #    self._save_models(path)
            #for tau in self.tau_points:
            #    self.model_dict[tau].to('cuda') #shift back to GPU after training
        elif save:
            self._load_models(path)
        else:
            warnings.warn('Model is initialized but not trained')

    def _save_submodel(self, tau, directory):
        t = time()
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, f'model_tau_{tau}.pt')
        torch.save(self.model_dict[tau].state_dict(), filepath)
        print(f'Model(tau={tau}) saved to {filepath}')
        print(f'Save time: {time()-t:.2f}s')


    def _save_models(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for tau, model in self.model_dict.items():
            filepath = os.path.join(directory, f'model_tau_{tau}.pt')
            torch.save(model.state_dict(), filepath)
        print(f'Models saved to {directory}')

    def _load_models(self, directory, tau_checkpoint=False):
        num_loaded = 0
        try:
            #for tau in self.tau_points:
            for tau in reversed(self.tau_points):
                #if tau < tau_checkpoint: #for checkpointing, 100 -> 1
                if tau_checkpoint and tau > tau_checkpoint: #1 -> 100 #problem here
                    continue
                model = self.model_dict[tau]
                filepath = os.path.join(directory, f'model_tau_{tau}.pt')
                model.load_state_dict(torch.load(filepath, weights_only=True))
                num_loaded += 1
            print(f'{num_loaded} models loaded from {directory}')
        except FileNotFoundError:
            raise FileNotFoundError(f'Model file {filepath} not found')
