"""
Score-based SDE implementations (VPSDE & VESDE)
- Provides:
    - marginal_prob(x0, t): returns (mean, std) of q(x_t | x_0)
    - diffusion_coeff(t): returns diffusion coefficient (vector) for t
    - sample_q(x0, t): samples x_t ~ q(x_t|x0) and returns (x_t, eps)
Author: Swapan Mallick, 29 September, 2025
"""
import torch as th


class VPSDE:
    """
    Variance-Preserving SDE (discrete-time schedule wrapper).
    The SDE used is analogous to the VP SDE from Song et al.
    This class expects 't' in [0, 1] (float or tensor). It maps t -> index in [0, N-1].
    """

    def __init__(self, N=1000, betas=None, loss_type="score_matching"):
        self.N = int(N)
        self.loss_type = loss_type

        # Convert betas to torch tensor (float32)
        if betas is None:
            self.betas = th.linspace(1e-4, 0.02, self.N, dtype=th.float32)
        elif not th.is_tensor(betas):
            self.betas = th.tensor(betas, dtype=th.float32)
        else:
            self.betas = betas.to(dtype=th.float32)

        # compute alphas and cumulative product as torch tensors
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(self.alphas, dim=0)

    def _time_to_index(self, t, device):
        # Accept scalar, list/ndarray, or torch tensor in [0,1]. Return long indices on device.
        if not th.is_tensor(t):
            t = th.tensor(t, dtype=th.float32, device=device)
        else:
            t = t.to(device=device, dtype=th.float32)
        idx = (t * (self.N - 1)).long().clamp(0, self.N - 1)
        return idx

    def marginal_prob(self, x0, t):
        """
        Return (mean, std) of q(x_t | x_0).
        x0: tensor [B, C, H, W]
        t: scalar or tensor in [0,1] with shape [B] or []
        Returns:
            mean: [B, C, H, W]
            std:  [B, 1, 1, 1]
        """
        device = x0.device
        idx = self._time_to_index(t, device)
        alphas_cumprod = self.alphas_cumprod.to(device)

        # index may be scalar or vector
        alpha_bar = alphas_cumprod[idx]  # shape [B] or scalar
        if alpha_bar.dim() == 0:
            alpha_bar = alpha_bar.unsqueeze(0)

        mean = th.sqrt(alpha_bar)[:, None, None, None] * x0
        std = th.sqrt(1.0 - alpha_bar)[:, None, None, None]
        return mean, std

    def diffusion_coeff(self, t):
        """
        Returns sqrt(beta_t) for given t (vector shape [B] or scalar).
        """
        # if t is tensor or scalar
        device = t.device if th.is_tensor(t) else th.device("cpu")
        idx = self._time_to_index(t, device)
        betas = self.betas.to(device)
        beta_t = betas[idx]
        return th.sqrt(beta_t)

    def sample_q(self, x0, t):
        """
        Sample x_t ~ q(x_t | x_0) given x0 and t in [0,1].
        Returns: (x_t, eps) where eps ~ N(0, I) used to generate x_t = mean + std * eps.
        """
        mean, std = self.marginal_prob(x0, t)
        eps = th.randn_like(x0)
        xt = mean + std * eps
        return xt, eps


class VESDE:
    """
    Variance-Exploding SDE wrapper (discrete sigma schedule).
    Uses sigma(t) schedule discretized over N steps. Works similarly to VPSDE API.
    """

    def __init__(self, N=1000, sigma_min=0.01, sigma_max=50.0, loss_type="score_matching"):
        self.N = int(N)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.loss_type = loss_type

        # discretized sigma schedule as torch tensor
        self.sigmas = th.exp(
            th.linspace(th.log(th.tensor(self.sigma_min)),
                        th.log(th.tensor(self.sigma_max)),
                        self.N,
                        dtype=th.float32)
        )

    def _time_to_index(self, t, device):
        if not th.is_tensor(t):
            t = th.tensor(t, dtype=th.float32, device=device)
        else:
            t = t.to(device=device, dtype=th.float32)
        idx = (t * (self.N - 1)).long().clamp(0, self.N - 1)
        return idx

    def marginal_prob(self, x0, t):
        """
        For VE SDE: p(x_t | x_0) = N(x_0, sigma(t)^2 I)
        Returns (mean, std)
        """
        device = x0.device
        idx = self._time_to_index(t, device)
        sigmas = self.sigmas.to(device)
        sigma_t = sigmas[idx]
        if sigma_t.dim() == 0:
            sigma_t = sigma_t.unsqueeze(0)
        mean = x0
        std = sigma_t[:, None, None, None]
        return mean, std

    def diffusion_coeff(self, t):
        """
        Return sigma(t) (vector of shape [B] or scalar).
        """
        device = t.device if th.is_tensor(t) else th.device("cpu")
        idx = self._time_to_index(t, device)
        sigmas = self.sigmas.to(device)
        sigma_t = sigmas[idx]
        return sigma_t

    def sample_q(self, x0, t):
        """
        Sample x_t ~ N(x0, sigma(t)^2 I).
        Returns (x_t, eps) where eps is standard normal.
        """
        mean, std = self.marginal_prob(x0, t)
        eps = th.randn_like(x0)
        xt = mean + std * eps
        return xt, eps
