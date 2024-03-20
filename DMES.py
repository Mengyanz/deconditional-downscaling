# Based on Botorch MES implementation
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from math import log
from typing import Any, Callable, Optional

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy, MaxValueBase, _sample_max_value_Thompson
from botorch.acquisition.cost_aware import CostAwareUtility, InverseCostWeightedUtility
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.model import Model
from botorch.models.utils import check_no_nans
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform

from linear_operator.functions import inv_quad
from linear_operator.utils.cholesky import psd_safe_cholesky
from scipy.optimize import brentq
from scipy.stats import norm
from torch import Tensor

import matplotlib.pyplot as plt
from gpytorch.distributions import MultivariateNormal


CLAMP_LB = 1.0e-8

class DeMaxValueBase(AcquisitionFunction, ABC):
    r"""Abstract base class for acquisition functions based on Max-value Entropy Search.

    This class provides the basic building blocks for constructing max-value
    entropy-based acquisition functions along the lines of [Wang2017mves]_.

    Subclasses need to implement `_sample_max_values` and _compute_information_gain`
    methods.

    :meta private:
    """

    def __init__(
        self,
        model: Model,
        num_mv_samples: int,
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""Single-outcome max-value entropy search-based acquisition functions.

        Args:
            model: A fitted single-outcome model.
            num_mv_samples: Number of max value samples.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
        """
        super().__init__(model=model)

        # if posterior_transform is None and model.num_outputs != 1:
        #     raise UnsupportedError(
        #         "Must specify a posterior transform when using a multi-output model."
        #     )

        # Batched GP models are not currently supported
        # try:
        #     batch_shape = model.batch_shape
        # except NotImplementedError:
        #     batch_shape = torch.Size()
        # if len(batch_shape) > 0:
        #     raise NotImplementedError(
        #         "Batched GP models (e.g., fantasized models) are not yet "
        #         f"supported by `{self.__class__.__name__}`."
        #     )
        self.num_mv_samples = num_mv_samples
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.weight = 1.0 if maximize else -1.0
        self.set_X_pending(X_pending)

    # @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute max-value entropy at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        # Compute the posterior, posterior mean, variance and std
        # posterior = self.model(X.unsqueeze(-3))
        posterior = self.model(X)
        # batch_shape x num_fantasies x (m) x 1
        mean = self.weight * posterior.mean.unsqueeze(-1)
        variance = posterior.variance.clamp_min(CLAMP_LB).view_as(mean)
        ig = self._compute_information_gain(
            X=X, mean_M=mean, variance_M=variance, covar_mM=variance.unsqueeze(-1)
        )
        return ig.mean(dim=0)  # average over fantasies

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending design points.

        Set "pending points" to inform the acquisition function of the candidate
        points that have been generated but are pending evaluation.

        Args:
            X_pending: `n x d` Tensor with `n` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is not None:
            X_pending = X_pending.detach().clone()
        self._sample_max_values(num_samples=self.num_mv_samples, X_pending=X_pending)
        self.X_pending = X_pending

    # ------- Abstract methods that need to be implemented by subclasses ------- #

    @abstractmethod
    def _compute_information_gain(self, X: Tensor, **kwargs: Any) -> Tensor:
        r"""Compute the information gain at the design points `X`.

        `num_fantasies = 1` for non-fantasized models.

         Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design point each.
            kwargs: Other keyword arguments used by subclasses.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X` (`num_fantasies=1` for non-fantasized models).
        """
        pass  # pragma: no cover

    @abstractmethod
    def _sample_max_values(
        self, num_samples: int, X_pending: Optional[Tensor] = None
    ) -> Tensor:
        r"""Draw samples from the posterior over maximum values.

        These samples are used to compute Monte Carlo approximations of expectations
        over the posterior over the function maximum.

        Args:
            num_samples: The number of samples to draw.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.

        Returns:
            A `num_samples x num_fantasies` Tensor of posterior max value samples
            (`num_fantasies=1` for non-fantasized models).
        """
        pass  # pragma: no cover


   
class DeDiscreteMaxValueBase(DeMaxValueBase):
    def __init__( self,
        model: Model,
        x_set: Tensor,
        num_mv_samples: int = 10,
        posterior_transform: Optional[PosteriorTransform] = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        train_inputs: Optional[Tensor] = None,
        normalise_para = [0,1]) -> None:
        self.use_gumbel = use_gumbel

        if train_inputs is None and hasattr(model, "train_inputs"):
            train_inputs = model.train_inputs[0]
        # if train_inputs is not None:
        #     if train_inputs.ndim > 2:
        #         raise NotImplementedError(
        #             "Batch GP models (e.g. fantasized models) "
        #             "are not yet supported by `MaxValueBase`"
        #         )
        #     train_inputs = match_batch_shape(train_inputs, candidate_set)
        #     candidate_set = torch.cat([candidate_set, train_inputs], dim=0)

        self.candidate_set = x_set
        self.normalise_para = normalise_para

        super().__init__(
            model=model,
            num_mv_samples=num_mv_samples,
            posterior_transform=posterior_transform,
            maximize=maximize,
            X_pending=X_pending,
        )
        
    def _sample_max_values(
            self, num_samples: int, X_pending: Optional[Tensor] = None
        ) -> Tensor:
        r"""Draw samples from the posterior over maximum values on a discrete set.

        These samples are used to compute Monte Carlo approximations of expectations
        over the posterior over the function maximum.

        Args:
            num_samples: The number of samples to draw.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.

        Returns:
            A `num_samples x num_fantasies` Tensor of posterior max value samples
            (`num_fantasies=1` for non-fantasized models).
        """
        if self.use_gumbel:
            sample_max_values = _sample_max_value_Gumbel
        else:
            sample_max_values = _sample_max_value_Thompson
        candidate_set = self.candidate_set

        with torch.no_grad():
            if X_pending is not None:
                # Append X_pending to candidate set
                X_pending = match_batch_shape(X_pending, self.candidate_set)
                candidate_set = torch.cat([self.candidate_set, X_pending], dim=0)

            # project the candidate_set to the highest fidelity,
            # which is needed for the multi-fidelity MES
            try:
                candidate_set = self.project(candidate_set)
            except AttributeError:
                pass

            self.posterior_max_values = sample_max_values(
                model=self.model,
                candidate_set=candidate_set,
                num_samples=self.num_mv_samples,
                posterior_transform=self.posterior_transform,
                maximize=self.maximize,
                normalise_para=self.normalise_para
            )
     
    
class qDeMaxValueEntropy(DeDiscreteMaxValueBase, MCSamplerMixin):
    def __init__(self,
        model: Model,
        x_set: Tensor,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 10, # TODO: 128
        posterior_transform: Optional[PosteriorTransform] = None,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        train_inputs: Optional[Tensor] = None,
        normalise_para = [0,1]) -> None:
        use_gumbel = False # TODO: test
        super().__init__(model=model,
            x_set = x_set,
            num_mv_samples=num_mv_samples,
            posterior_transform=posterior_transform,
            use_gumbel=use_gumbel,
            maximize=maximize,
            X_pending=X_pending,
            train_inputs=train_inputs,
            normalise_para=normalise_para)
        
        MCSamplerMixin.__init__(
            self,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([num_y_samples])),
        )
        self._init_model = model  # used for `fantasize()` when setting `X_pending`
        self.fantasies_sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([num_fantasies])
        )
        self.num_fantasies = num_fantasies
        self.set_X_pending(X_pending)  # this did not happen in the super constructor

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending points.

        Informs the acquisition function about pending design points,
        fantasizes the model on the pending points and draws max-value samples
        from the fantasized model posterior.

        Args:
            X_pending: `m x d` Tensor with `m` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        try:
            init_model = self._init_model
        except AttributeError:
            # Short-circuit (this allows calling the super constructor)
            return
        if X_pending is not None:
            # fantasize the model and use this as the new model
            self.model = init_model.fantasize(
                X=X_pending,
                sampler=self.fantasies_sampler,
            )
        else:
            self.model = init_model
        super().set_X_pending(X_pending)

    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        r"""Computes the information gain at the design points `X`.

        Approximately computes the information gain at the design points `X`,
        for both MES with noisy observations and multi-fidelity MES with noisy
        observation and trace observations.

        The implementation is inspired from the papers on multi-fidelity MES by
        [Takeno2020mfmves]_. The notation in the comments in this function follows
        the Appendix C of [Takeno2020mfmves]_.

        `num_fantasies = 1` for non-fantasized models.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design point each.
            mean_M: A `batch_shape x num_fantasies x (m)`-dim Tensor of means.
            variance_M: A `batch_shape x num_fantasies x (m)`-dim Tensor of variances.
            covar_mM: A
                `batch_shape x num_fantasies x (m) x (1 + num_trace_observations)`-dim
                Tensor of covariances.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X` (`num_fantasies=1` for non-fantasized models).
        """
        # compute the std_m, variance_m with noisy observation
        # posterior_m = self.model(X.unsqueeze(-3))
        # TODO: check whether this gives the posterior prediction of z
        posterior_m = self.model(X)
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        mean_m = self.weight * posterior_m.mean.unsqueeze(-1).unsqueeze(-1)
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        variance_m = (posterior_m.stddev ** 2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        
        check_no_nans(variance_m)

        # compute mean and std for fM|ym, x, Dt ~ N(u, s^2)
        # TODO: we need posterior for z instead of f -- need to be fixed!!
        # samples_m = self.weight * self.get_posterior_samples(posterior_m).squeeze(-1)
        samples_m = torch.stack([posterior_m.sample() for i in range(128)]).unsqueeze(-1).unsqueeze(-1)
        # print('samples_m ', samples_m.shape)
        # print(samples_m)
        # print('mean_m: ', mean_m.shape)
        # print('variance_m: ', variance_m.shape)
        # print('samples_m - mean_m: ', (samples_m - mean_m).shape)
        
        # s_m x batch_shape x num_fantasies x (m) (1 + num_trace_observations)
        L = psd_safe_cholesky(variance_m)
        # print('covar_mM shape: ', covar_mM.shape)
        # print('mean_M shape: ', mean_M.shape)
        # print('variance_M shape: ',variance_M.shape)
        # print('L shape: ', L.shape)
        temp_term = torch.cholesky_solve(covar_mM.unsqueeze(-1), L).transpose(-2, -1)
        # equivalent to torch.matmul(covar_mM.unsqueeze(-2), torch.inverse(variance_m))
        # batch_shape x num_fantasies (m) x 1 x (1 + num_trace_observations)

        mean_pt1 = torch.matmul(temp_term, (samples_m - mean_m).unsqueeze(-1))
        mean_new = mean_pt1.squeeze(-1).squeeze(-1) + mean_M
        # s_m x batch_shape x num_fantasies x (m)
        variance_pt1 = torch.matmul(temp_term, covar_mM.unsqueeze(-1))
        variance_new = variance_M - variance_pt1.squeeze(-1).squeeze(-1)
        # batch_shape x num_fantasies x (m)
        stdv_new = variance_new.clamp_min(CLAMP_LB).sqrt()
        # batch_shape x num_fantasies x (m)

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # Compute p(fM <= f* | ym, x, Dt)
        view_shape = torch.Size(
            [
                self.posterior_max_values.shape[0],
                # add 1s to broadcast across the batch_shape of X
                *[1 for _ in range(X.ndim - self.posterior_max_values.ndim)],
                *self.posterior_max_values.shape[1:],
            ]
        )  # s_M x batch_shape x num_fantasies x (m)
        max_vals = self.posterior_max_values.view(view_shape).unsqueeze(1).unsqueeze(1)
        # print('max_vals: ', max_vals)
        # print('max_vals: ', max_vals.shape)
        # print('mean_new: ', mean_new.shape)
        # print('stdv_vals: ', stdv_new.shape)
        # s_M x 1 x batch_shape x num_fantasies x (m)
        normalized_mvs_new = (max_vals - mean_new) / stdv_new
        # s_M x s_m x batch_shape x num_fantasies x (m)  =
        #   s_M x 1 x batch_shape x num_fantasies x (m)
        #   - s_m x batch_shape x num_fantasies x (m)
        cdf_mvs_new = normal.cdf(normalized_mvs_new).clamp_min(CLAMP_LB)

        # Compute p(fM <= f* | x, Dt)
        stdv_M = variance_M.sqrt()
        normalized_mvs = (max_vals - mean_M) / stdv_M
        # s_M x 1 x batch_shape x num_fantasies  x (m) =
        # s_M x 1 x 1 x num_fantasies x (m) - batch_shape x num_fantasies x (m)
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        # s_M x 1 x batch_shape x num_fantasies x (m)

        # Compute log(p(ym | x, Dt))
        # log_pdf_fm = posterior_m.log_prob(
        #     self.weight * samples_m
        # ).unsqueeze(0)
        log_pdf_fm = torch.stack([-0.5 * torch.log(2 * torch.pi * variance_m.squeeze(-1)) - 0.5 * torch.div(torch.square(samples_m[i] - mean_m), variance_m.squeeze(-1)) for i in range(samples_m.shape[0])]).squeeze(-1).unsqueeze(0)
        # print('log_pdf_fm: ', log_pdf_fm.shape)
        # print(log_pdf_fm)
        # 1 x s_m x batch_shape x num_fantasies x (m)

        # H0 = H(ym | x, Dt)
        # H0 = posterior_m.entropy()  # batch_shape x num_fantasies x (m)
        H0 = torch.log(torch.sqrt(2 * torch.pi * torch.e * variance_m)).squeeze(-1).squeeze(-1)
        # print('H0: ', H0.shape)

        # regression adjusted H1 estimation, H1_hat = H1_bar - beta * (H0_bar - H0)
        # H1 = E_{f*|x, Dt}[H(ym|f*, x, Dt)]
        Z = cdf_mvs_new / cdf_mvs  # s_M x s_m x batch_shape x num_fantasies x (m)
        # s_M x s_m x batch_shape x num_fantasies x (m)
        h1 = -Z * Z.log() - Z * log_pdf_fm
        check_no_nans(h1)
        dim = [0, 1]  # dimension of fm samples, fM samples
        H1_bar = h1.mean(dim=dim)
        h0 = -log_pdf_fm
        H0_bar = h0.mean(dim=dim)
        cov = ((h1 - H1_bar) * (h0 - H0_bar)).mean(dim=dim)
        beta = cov / (h0.var(dim=dim) * h1.var(dim=dim)).sqrt()
        H1_hat = H1_bar - beta * (H0_bar - H0)
        ig = H0 - H1_hat  # batch_shape x num_fantasies x (m)
        
        # fig = plt.figure()
        # plt.plot(X.squeeze(-1).detach().numpy() , H1_bar.squeeze(-1).detach().numpy() , label = 'H1_bar')
        # plt.plot(X.squeeze(-1).detach().numpy(), (H0_bar - H0).squeeze(-1).detach().numpy(), label = 'H0_bar - H0')
        # plt.plot(X.squeeze(-1).detach().numpy(), beta.squeeze(-1).detach().numpy(), label = 'beta') 
        # plt.title('H1 objective function')
        # plt.legend()
        # plt.show()
        # plt.close(fig) 
        
        # print('H0 shape: ', H0.shape)
        # print('H1 shape: ', H1_hat.shape)
        # print('IG shape: ', ig.shape)
        fig = plt.figure()
        # plt.plot(X.squeeze(-1).detach().numpy() , H0.squeeze(-1).detach().numpy() , label = 'H0')
        # plt.plot(X.squeeze(-1).detach().numpy(), H1_hat.squeeze(-1).detach().numpy(), label = 'H1')
        plt.plot(X.squeeze(-1).detach().numpy(), ig.squeeze(-1).detach().numpy(), label = 'IG') 
        # plt.ylim(0,1)
        plt.title('MES objective function')
        plt.legend()
        plt.show()
        plt.close(fig) 
        
        if self.posterior_max_values.ndim == 2:
            permute_idcs = [-1, *range(ig.ndim - 1)]
        else:
            permute_idcs = [-2, *range(ig.ndim - 2), -1]
        ig = ig.permute(*permute_idcs)  # num_fantasies x batch_shape x (m)
        
        return ig
    
    
def _sample_max_value_Gumbel(
    model: Model,
    candidate_set: Tensor,
    num_samples: int,
    posterior_transform: Optional[PosteriorTransform] = None,
    maximize: bool = True,
) -> Tensor:
    """Samples the max values by Gumbel approximation.

    Should generally be called within a `with torch.no_grad()` context.

    Args:
        model: A fitted single-outcome model.
        candidate_set: A `n x d` Tensor including `n` candidate points to
            discretize the design space.
        num_samples: Number of max value samples.
        posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A `num_samples x num_fantasies` Tensor of posterior max value samples.
    """
    # define the approximate CDF for the max value under the independence assumption
    posterior = model.predict(candidate_set)
    weight = 1.0 if maximize else -1.0
    mu = weight * posterior.mean.unsqueeze(-1)
    sigma = posterior.variance.clamp_min(1e-8).sqrt().view_as(mu)
    # mu, sigma is (num_fantasies) X n X 1
    if len(mu.shape) == 3 and mu.shape[-1] == 1:
        mu = mu.squeeze(-1).T
        sigma = sigma.squeeze(-1).T
    # mu, sigma is now n X num_fantasies or n X 1

    # bisect search to find the quantiles 25, 50, 75
    lo = (mu - 3 * sigma).min(dim=0).values
    hi = (mu + 5 * sigma).max(dim=0).values
    num_fantasies = mu.shape[1]
    device = candidate_set.device
    dtype = candidate_set.dtype
    quantiles = torch.zeros(num_fantasies, 3, device=device, dtype=dtype)
    for i in range(num_fantasies):
        lo_, hi_ = lo[i], hi[i]
        N = norm(mu[:, i].cpu().numpy(), sigma[:, i].cpu().numpy())
        quantiles[i, :] = torch.tensor(
            [
                brentq(lambda y: np.exp(np.sum(N.logcdf(y))) - p, lo_, hi_)
                for p in [0.25, 0.50, 0.75]
            ]
        )
    q25, q50, q75 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    # q25, q50, q75 are 1 dimensional tensor with size of either 1 or num_fantasies

    # parameter fitting based on matching percentiles for the Gumbel distribution
    b = (q25 - q75) / (log(log(4.0 / 3.0)) - log(log(4.0)))
    a = q50 + b * log(log(2.0))

    # inverse sampling from the fitted Gumbel CDF distribution
    sample_shape = (num_samples, num_fantasies)
    eps = torch.rand(*sample_shape, device=device, dtype=dtype)
    max_values = a - b * eps.log().mul(-1.0).log()

    return max_values  # num_samples x num_fantasies


def _sample_max_value_Thompson(
    model: Model,
    candidate_set: Tensor,
    num_samples: int,
    posterior_transform: Optional[PosteriorTransform] = None,
    maximize: bool = True,
    normalise_para = [0,1]
) -> Tensor:
    """Samples the max values by discrete Thompson sampling.

    Should generally be called within a `with torch.no_grad()` context.

    Args:
        model: A fitted single-outcome model.
        candidate_set: A `n x d` Tensor including `n` candidate points to
            discretize the design space.
        num_samples: Number of max value samples.
        posterior_transform: A PosteriorTransform. If using a multi-output model,
            a PosteriorTransform that transforms the multi-output posterior into a
            single-output posterior is required.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A `num_samples x num_fantasies` Tensor of posterior max value samples.
    """
    print('_sample_max_value_Thompson')
    posterior = model.predict(candidate_set)
    muz, sigmaz = normalise_para
    print('normalise muz: ', muz)
    print('normalise sigmaz: ', sigmaz)
    # mean = sigmaz * posterior.mean + muz
    # covariance_matrix = sigmaz ** 2 * posterior.covariance_matrix # not positive definite
    # posterior = MultivariateNormal(mean = mean, covariance_matrix=covariance_matrix)
    
    weight = 1.0 if maximize else -1.0
    samples = weight * posterior.rsample(torch.Size([num_samples])).squeeze(-1) + muz
    # TODO: figure out how to rescale samples, only adding muz for now
    # samples is num_samples x (num_fantasies) x n
   
    max_values, _ = samples.max(dim=-1)
    if len(samples.shape) == 2:
        max_values = max_values.unsqueeze(-1)  # num_samples x num_fantasies
        
    fig = plt.figure()
    candidate_set=candidate_set.reshape(-1,)
    mean = sigmaz * posterior.mean + muz
    std = sigmaz * posterior.stddev
    plt.plot(candidate_set, mean, label = 'posterior mean', color='C0')
    plt.fill_between(candidate_set, mean - 2 * std, mean + 2 * std, alpha=0.3, color='C0')
    # plt.plot(groundtruth_individuals, (conf[1]-conf[0]).detach().numpy(), label = '2 * posterior std')
    # plt.plot(self.x_space, self.f_oracle(self.x_space), label = 'f(x)', color='C1')
    plt.xlim(candidate_set[0], candidate_set[-1])
    # plt.ylim(-5, 5)
    # plt.plot(xs, f(xs), '.', label = 'data points')
    for i, sample in enumerate(samples[:10]):
        plt.plot(candidate_set, sample, alpha = 0.5)
        plt.axhline(y=max_values[i], alpha = 0.5)
    # plt.plot(self.y_recs, self.z_rewards, '.', label = 'observations z')
    plt.title('Posterior f and samples, max values via TS')
    plt.legend()
    # plt.savefig(f"{self.dump_dir}predf/frame_{num_exp}_{num_iter}.png")
    plt.show()
    plt.close(fig)

    return max_values