import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
from tqdm import tqdm
from models.losses import CharbonnierLoss

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDPM(nn.Module):
    def __init__(self, model, img_channels, betas, criterion='l1', device='cuda'):
        super().__init__()
        self.model = nn.DataParallel(model).to(device)
        self.img_channels = img_channels
        self.num_timesteps = len(betas)
        if criterion == 'l1':
            self.criterion = CharbonnierLoss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("loss criterion must be l1 or l2")
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod",
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas",
                             to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(
            betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(self, x, condition, t):
        input = torch.cat([x, condition], dim=1) 
        return ((x - extract(self.remove_noise_coeff, t, x.shape) * self.model(input, t)) * extract(self.reciprocal_sqrt_alphas, t, x.shape))

    @torch.no_grad()
    def sample(self, condition, device, tqdm_visible=False):
        b, c, h, w = condition.shape
        x = torch.randn((b, 3, h, w), device=device)

        if tqdm_visible:
            timesteps_list = tqdm(range(self.num_timesteps - 1, -1, -1), desc='sampling loop time step', total=self.num_timesteps)
        else:
            timesteps_list = range(self.num_timesteps - 1, -1, -1)

        for t in timesteps_list:
            t_batch = torch.tensor([t], device=device).repeat(b)
            x = self.remove_noise(x, condition, t_batch)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * \
                    torch.randn_like(x)

        return x.cpu().detach()

    def perturb_x(self, x, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x.shape) * x + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise)

    def compute_loss(self, x, condition, t):
        noise = torch.randn_like(x)
        pred = self.perturb_x(x, t, noise)
        input = torch.cat([pred, condition], dim=1)
        pred_noise = self.model(input, t)
        return self.criterion(pred_noise, noise)

    def forward(self, x, condition):
        b, c, h, w = x.shape
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.compute_loss(x, condition, t)


class DDIM(DDPM):
    @torch.no_grad()
    def sample(self, condition, sample_timesteps=20, ddim_eta=0.0, device="cuda", tqdm_visible=False, init_noise=None):
        b, c, h,w = condition.shape
        ddim_timestep_seq = np.asarray(
            list(range(0, self.num_timesteps, self.num_timesteps // sample_timesteps)))
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(
            np.array([0]), ddim_timestep_seq[:-1])
        if init_noise is not None:
            x = init_noise
        else:
            x = torch.randn((b, 3, h, w), device=device)
        if tqdm_visible:
            timesteps_list = tqdm(reversed(range(0, sample_timesteps)), desc='sampling loop time step', total=sample_timesteps)
        else:
            timesteps_list = reversed(range(0, sample_timesteps))
        for i in timesteps_list:
            t_batch = torch.tensor(
                [ddim_timestep_seq[i]], device=device, dtype=torch.long).repeat(b)
            prev_t_batch = torch.tensor(
                [ddim_timestep_prev_seq[i]], device=device, dtype=torch.long).repeat(b)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(
                self.alphas_cumprod, t_batch, x.shape)
            alpha_cumprod_t_prev = extract(
                self.alphas_cumprod, prev_t_batch, x.shape)

            # 2. predict noise using model
            input = torch.cat([x, condition], dim=1)
            pred_noise = self.model(input, t_batch)

            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t))
                    * pred_noise) / torch.sqrt(alpha_cumprod_t)

            # 4. compute variance: "sigma_t(η)"
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t"
            pred_dir_xt = torch.sqrt(
                1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise

            # 6. compute x_{t-1}
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + \
                pred_dir_xt + sigmas_t * torch.randn_like(x)

            x = x_prev

        return x.cpu().detach()


class EMA():
    def __init__(self, decay):
        self.decay = decay

    def __call__(self, old, new):
        old_dict = old.state_dict()
        new_dict = new.state_dict()
        for key in old_dict.keys():
            new_dict[key].data = old_dict[key].data * \
                self.decay + new_dict[key].data * (1 - self.decay)