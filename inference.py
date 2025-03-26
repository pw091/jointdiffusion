import torch

from pi_ddpm_aux import *
from pi_ddpm_helpers import *

def _infer_single(model, x_init, t_init, n_steps, dt, ensemble_size):
    x_pred_arr = np.empty((model.data_dim, n_steps + model.n_joint_cond))
    x_pred_arr[:, 0:model.n_joint_cond] = x_init

    x = x_init.T.flatten()
    for t_idx in range(n_steps):
        if t_idx == 1:
            start_time = time()
        elif t_idx == 2:
            print(eta_format((time()-start_time) * n_steps))
        elif t_idx == 11:
            print(eta_format((time()-start_time) * n_steps/10))
        elif t_idx % 50 == 0:
            print(f'{t_idx}/{n_steps}')

        # t_next = t_init + (t_idx + 1) * dt  # if not self.is_shifted else t_init + (t_idx)*dt
        #x_pred_ensemble = model.infer(torch.tensor(x).float(), t_init + t_idx * dt, ensemble_size)  # (ensemble_size, dim)
        x_pred_ensemble = model.infer(torch.tensor(x).float(), ensemble_size)
        x_pred = torch.mean(x_pred_ensemble, dim=0).cpu().detach().numpy()
        x_pred_arr[:, model.n_joint_cond + t_idx] = x_pred
        # x = x_pred
        x = np.concatenate((x[model.data_dim:], x_pred), axis=0)

    return x_pred_arr

def _apply_featurizer(feat, flathead, tail):
    assert tail.shape[2] == feat.data_dim
    bsz = flathead.shape[0]
    head = torch.tensor(flathead).view(bsz, -1, feat.data_dim)
    tail = torch.tensor(tail).repeat(bsz, 1, 1)
    seq = torch.cat((tail, head), dim=1).float() #(bsz, input_len, data_dim)
    h = feat.encode(seq) #(bsz, latent_len, data_dim)
    return h.view(bsz, -1).cpu().detach().numpy() #(bsz, latent_len*data_dim)

def _infer_joint(model, x_init, t_init, n_steps, dt, cloud_size, chunk_size=None, featurizer=None):
    dim = x_init.shape[0]
    #torch
    x_init = torch.tensor(x_init).float()
    pred_traj = torch.zeros((dim, n_steps + model.n_joint_cond))
    #np
    #pred_traj = np.zeros((dim, n_steps + model.n_joint_cond))
    #pred_traj[:, :model.n_joint_cond] = x_init #proper for conditional
    pred_traj[:, :model.n_joint_out] = x_init #hacky workaround for unconditional
    x_prev_flat = x_init.T.flatten()

    if chunk_size is None:
        chunk_size = cloud_size
    assert cloud_size == chunk_size #torch
    assert cloud_size % chunk_size == 0 and cloud_size >= chunk_size
    if chunk_size < cloud_size:
        print(f'Chunking enabled: {chunk_size}/{cloud_size}')

    if model.n_joint_cond == 0:
        x_joint_cloud = model.infer(x_prev_flat, cloud_size, t_init * dt, dt)
        #x_prev_preds = x_joint_cloud[:, :dim * (model.n_joint_out - 1)]  # (t-n_joint+1, ..., t-1)
        #x_t_preds = x_joint_cloud[:, dim * (model.n_joint_out - 1):]  # (t)
        #x_prev_overlap = x_prev_flat[dim * (model.n_joint_cond - model.n_joint_out + 1):]  # exclude x(t-n_joint) since not in x_prev_preds

    for t_idx_step in range(n_steps):
        if t_idx_step == 1:
            start_time = time()
        elif t_idx_step == 2:
            print(eta_format((time()-start_time) * n_steps))

        if model.n_joint_cond >= 1:
            #x_joint_cloud = model.infer(torch.tensor(x_prev_flat).float(), t_init + t_idx_step * dt, cloud_size).cpu().detach().numpy()  # (t-n_joint+1, ..., t)
            #torch
            x_joint_cloud = model.infer(x_prev_flat, cloud_size, t_init + t_idx_step * dt, dt)
            #np
            #x_joint_cloud = model.infer(torch.tensor(x_prev_flat).float(), chunk_size).cpu().detach().numpy() if chunk_size == cloud_size else \
            #    np.concat([model.infer(torch.tensor(x_prev_flat).float(), chunk_size).cpu().detach().numpy() for _ in range(cloud_size//chunk_size)])
        else:
            x_joint_cloud = torch.clone(x_joint_cloud)

        x_prev_preds = x_joint_cloud[:, :dim * (model.n_joint_out - 1)]  # (t-n_joint+1, ..., t-1)
        x_t_preds = x_joint_cloud[:, dim * (model.n_joint_out - 1):]  # (t)
        x_prev_overlap = x_prev_flat[dim * (model.n_joint_cond - model.n_joint_out + 1):]  # exclude x(t-n_joint) since not in x_prev_preds

        #weights = np.ones(dim*(model.n_joint_out-1))
        #x_prev_preds = x_prev_preds * weights
        #x_prev_overlap = x_prev_overlap * weights

        if featurizer and t_idx_step >= featurizer.input_len - model.n_joint_cond:
            tail_idx = model.n_joint_cond + t_idx_step - featurizer.input_len #e.g. 10 + 90 - 100 = 0, 10 + 91 - 100 = 1, ...

            tail_overlap = np.expand_dims(pred_traj[:, tail_idx:tail_idx+featurizer.input_len-model.n_joint_cond+1], axis=0).transpose(0, 2, 1) #(1, 100, 3)
            h_overlap = _apply_featurizer(featurizer, np.expand_dims(x_prev_overlap, axis=0), tail_overlap).squeeze(0)
            h_overlap = 10000*h_overlap
            x_prev_overlap = np.concat((h_overlap, x_prev_overlap))

            tail_pred = np.expand_dims(pred_traj[:, tail_idx:tail_idx+featurizer.input_len-model.n_joint_cond+1], axis=0).transpose(0, 2, 1) #(1, 100, 3)
            h_pred = _apply_featurizer(featurizer, x_prev_preds, tail_pred)
            h_pred = 10000*h_pred
            x_prev_preds = np.concat((h_pred, x_prev_preds), axis=1)

        delta = 1e-1
        while True:
            if delta > 1e5:
                raise ValueError('Delta too large')
            if delta < 1e-8:
                break
                # print(f'Num valid @ delta={delta}: {num_valid}')
                # raise ValueError('Delta too small')

            #torch
            within_delta_indices = torch.norm(x_prev_preds - x_prev_overlap, dim=1) < delta
            #np
            #within_delta_indices = np.linalg.norm(x_prev_preds - x_prev_overlap, axis=1) < delta
            x_t_valid = x_t_preds[within_delta_indices]
            num_valid = len(x_t_valid)

            if num_valid < 1:
                delta *= 1.1
            elif num_valid > 1:
                x_t_preds = x_t_preds[within_delta_indices]
                x_prev_preds = x_prev_preds[within_delta_indices]
                delta *= 0.9
            else:
                break

        if t_idx_step % 50 == 0:
            print(f'{t_idx_step}, num valid @ delta={delta}: {num_valid}')
        #x_t = np.mean(x_t_valid, axis=0)
        x_t = x_t_valid.squeeze(0)
        #torch
        x_prev_flat = torch.cat((x_prev_flat[dim:], x_t), axis=0)
        #np
        #x_prev_flat = np.concatenate((x_prev_flat[dim:], x_t), axis=0)
        pred_traj[:, model.n_joint_cond + t_idx_step] = x_t

    return pred_traj.cpu().detach().numpy() #torch/np

def _infer_joint_std(model, x_init, t_init, n_steps, dt, cloud_size,
                     chunk_size=None, featurizer=None):
    """
    Inference function that:
      - Runs a step-by-step 'joint' inference (model.infer(...))
      - At each step, obtains the entire unfiltered x_t cloud (size=cloud_size).
      - Picks a single 'final' x_t (mean over the unfiltered ensemble).
    Returns:
      pred_traj:   (dim, n_steps + model.n_joint_cond)
      ens_clouds:  (n_steps, cloud_size, dim)

    *No filtering or subcloud logic here!*
    """
    dim = x_init.shape[0]

    # Convert initial input to torch
    x_init = torch.tensor(x_init, dtype=torch.float32)
    # This will store the final single trajectory
    pred_traj = torch.zeros((dim, n_steps + model.n_joint_cond))
    pred_traj[:, :model.n_joint_cond] = x_init

    x_prev_flat = x_init.T.flatten()  # shape=(dim*n_joint_cond,)

    if chunk_size is None:
        chunk_size = cloud_size
    assert cloud_size == chunk_size, "Currently require chunk_size==cloud_size"

    # We'll store the entire unfiltered cloud of x_t in a list or array
    ens_clouds_list = []  # will become shape (n_steps, cloud_size, dim)

    for t_idx_step in range(n_steps):
        if t_idx_step == 0:
            start_time = time()
        elif t_idx_step == 1:
            elapsed_est = (time() - start_time) * (n_steps)
            print(f"Estimated total inference time: {elapsed_est:.2f} s")

        # 1) Generate a cloud of shape (cloud_size, dim*(model.n_joint_out)).
        #    That includes x_(t-n_joint+1) ... x_t
        #    We'll take x_t_preds = the last chunk for each ensemble member
        x_joint_cloud = model.infer(
            x_prev_flat, cloud_size,
            t_init + t_idx_step * dt, dt
        )  # shape (cloud_size, dim*model.n_joint_out)

        # 2) The final "x_t_preds" block is the last portion:
        #    We'll assume it's shape (cloud_size, dim).
        x_t_preds = x_joint_cloud[:, dim * (model.n_joint_out - 1):]

        # 3) Store the entire x_t_preds in our ensemble list:
        #    This is the "unfiltered" ensemble for step t_idx_step
        #    shape = (cloud_size, dim)
        ens_clouds_list.append(x_t_preds.clone().cpu())

        # 4) For the "single final x_t", let's do a mean across ensemble
        x_t_mean = torch.mean(x_t_preds, dim=0)

        # 5) Slide window: drop the oldest chunk from x_prev_flat, then append x_t_mean
        x_prev_flat = torch.cat((x_prev_flat[dim:], x_t_mean), dim=0)

        # 6) Insert x_t_mean into pred_traj
        pred_traj[:, model.n_joint_cond + t_idx_step] = x_t_mean

    # Convert ensemble list to a single NumPy array of shape (n_steps, cloud_size, dim)
    ens_clouds = torch.stack(ens_clouds_list, dim=0).numpy()

    return pred_traj.detach().cpu().numpy(), ens_clouds

def infer_traj(model, x_init, t_init, n_steps, dt, ensemble_size, chunk_size=None, featurizer=None):
    print('infer traj called @ ', time())
    if model.n_joint_out == 1:
        return _infer_single(model, x_init, t_init, n_steps, dt, ensemble_size)
    return _infer_joint(model, x_init, t_init, n_steps, dt, ensemble_size, chunk_size, featurizer)
    #return _infer_joint_std(model, x_init, t_init, n_steps, dt, ensemble_size, chunk_size, featurizer)