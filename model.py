import torch.nn as nn
import torch
from torch import Tensor
import math
from unet import Unet
from tqdm import tqdm

class Diffuser(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8]):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size

        betas = self._cosine_variance_schedule(timesteps) # or self._linear_variance_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.-alphas_cumprod))

        self.model = Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    def forward(self, x, noise):
        # x:NCHW
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(x_t, t)

        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples: int, device="cuda") -> Tensor:
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
        for i in range(self.timesteps - 1, -1, -1):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            x_t = self._reverse_diffusion_with_clip(x_t, t, noise)

        x_t=(x_t + 1.) / 2. #[-1,1] to [0,1]

        return x_t
    
    def _cosine_variance_schedule(self, timesteps: int, epsilon=0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas
    
    def _linear_variance_schedule(self, timesteps: int):
        '''
            generate cosine variance schedule
            reference: the DDPM paper https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
            You might compare the model performance of linear and cosine variance schedules. 
        '''
        raise NotImplementedError
        # ---------- **** ---------- #
        # YOUR CODE HERE
        betas = ...
        return betas
        # ---------- **** ---------- #

    def _forward_diffusion(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        '''
            forward diffusion process
            hint: calculate x_t given x_0, t, noise
            please note that alpha related tensors are registered as buffers in __init__, you can use gather method to get the values
            reference: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process
        '''
        raise NotImplementedError
        # ---------- **** ---------- #
        # YOUR CODE HERE
        x_t = ...
        return x_t
        # ---------- **** ---------- #


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor: 
        '''
            reverse diffusion process with clipping
            hint: with clip: pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
                  without clip: pred_noise -> pred_mean and pred_std
                  you may compare the model performance with and without clipping
        '''
        pred=self.model(x_t,t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 
    