import matplotlib.pyplot as plt
from sequential_tau.model_seq import *

class EpsilonThetaBase(EpsilonTheta):
    def __init__(self, sequence, tau):
        super(EpsilonThetaBase, self).__init__(sequence, tau)
    
    def forward(self, x_cond, x_tau):
        return self._forward_std(x_cond, x_tau)

    def train(self, data, t_points, bsz_s, bsz_t, n_iters, optimizer, scheduler):
        return self._train_std(data, t_points, bsz_s, bsz_t, n_iters, optimizer, scheduler)

class DDPMSequence(ModelSequence):
    def __init__(self, tau_points, data_dim, n_joint_cond, n_joint_out, loss, out_mode, cond_lag, is_time_dependent, dmodel, nheads, nlayers, dff, posfreqs, vram_saver):
        super(DDPMSequence, self).__init__(tau_points, data_dim, n_joint_cond, n_joint_out, loss, out_mode, cond_lag, is_time_dependent, dmodel, nheads, nlayers, dff, posfreqs, vram_saver)
        for tau in tau_points:
            self.model_dict[tau] = EpsilonThetaBase(self, tau)
            #if self.vram_saver:
            #    self.model_dict[tau].to('cpu') #model at each time step