import os
import yaml
import logging
import torch
import gpytorch
import matplotlib.pyplot as plt
from progress.bar import Bar
from models import VariationalCMP, CMPLikelihood, BagVariationalELBO, RFFKernel, MODELS, TRAINERS, PREDICTERS
from core.visualization import plot_downscaling_prediction
from core.metrics import compute_metrics
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


@MODELS.register('variational_cmp')
def build_downscaling_variational_cmp(covariates_grid, lbda, n_inducing_points, use_individuals_noise, **kwargs):
    """Hard-coded initialization of Variational CME Process module used for downscaling experiment

    Args:
        individuals (torch.Tensor)
        lbda (float)
        n_inducing_points (int)
        seed (int)

    Returns:
        type: VariationalCMP

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    individuals_mean = gpytorch.means.ZeroMean()

    # Define individuals kernel
    base_indiv_spatial_kernel = RFFKernel(nu=1.5, num_samples=1000, ard_num_dims=2, active_dims=[0, 1])
    base_indiv_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=2))

    base_indiv_feat_kernel = gpytorch.kernels.RFFKernel(num_samples=1000, ard_num_dims=3, active_dims=[2, 3, 4])
    base_indiv_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    individuals_spatial_kernel = gpytorch.kernels.ScaleKernel(base_indiv_spatial_kernel)
    individuals_feat_kernel = gpytorch.kernels.ScaleKernel(base_indiv_feat_kernel)
    individuals_kernel = individuals_spatial_kernel + individuals_feat_kernel

    # Define bags kernels
    base_bag_spatial_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1])
    base_bag_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=2))

    base_bag_feat_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[2])
    base_bag_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=1))

    bag_spatial_kernel = gpytorch.kernels.ScaleKernel(base_bag_spatial_kernel)
    bag_feat_kernel = gpytorch.kernels.ScaleKernel(base_bag_feat_kernel)
    bag_kernel = bag_spatial_kernel + bag_feat_kernel

    # Initialize inducing points regularly across grid
    flattened_grid = covariates_grid.view(-1, covariates_grid.size(-1))
    n_samples = flattened_grid.size(0)
    step = n_samples // n_inducing_points
    offset = (n_samples % n_inducing_points) // 2
    inducing_points = flattened_grid[offset:n_samples - offset:step].float()

    # Define model
    model = VariationalCMP(individuals_mean=individuals_mean,
                           individuals_kernel=individuals_kernel,
                           bag_kernel=bag_kernel,
                           inducing_points=inducing_points,
                           lbda=lbda,
                           use_individuals_noise=use_individuals_noise)
    return model


@TRAINERS.register('variational_cmp')
def train_downscaling_variational_cmp(model, covariates_blocks, bags_blocks, extended_bags, targets_blocks, batch_size_cme,
                                      lr, n_epochs, batch_size, beta, seed, dump_dir, device_idx, covariates_grid, missing_bags_fraction,
                                      use_individuals_noise, groundtruth_field, target_field, plot, plot_every, log_every, **kwargs):
    """Hard-coded training script of Exact CME Process for downscaling experiment

    Args:
        model (VariationalGP)
        covariates_blocks (torch.Tensor)
        bags_blocks (torch.Tensor)
        extended_bags (torch.Tensor)
        targets_blocks (torch.Tensor)
        lr (float)
        n_epochs (int)
        beta (float)
        batch_size (int)
        seed (int)
        dump_dir (str)
        covariates_grid (torch.Tensor)
        step_size (int)
        groundtruth_field (xarray.core.dataarray.DataArray)
        target_field (torch.Tensor)
        plot (bool)

    """
    # Transfer on device
    device = torch.device(f"cuda:{device_idx}") if torch.cuda.is_available() else torch.device("cpu")
    covariates_grid = covariates_grid.to(device)
    covariates_blocks = covariates_blocks.to(device)
    bags_blocks = bags_blocks.to(device)
    extended_bags = extended_bags.to(device)
    targets_blocks = targets_blocks.to(device)

    # Split dataset in unmatched sets
    if seed:
        torch.random.manual_seed(seed)
    n_drop = int(missing_bags_fraction * len(targets_blocks))
    shuffled_indices = torch.randperm(len(targets_blocks)).to(device)
    indices_1, indices_2 = shuffled_indices[n_drop:], shuffled_indices[:n_drop]

    covariates_blocks = covariates_blocks[indices_1].reshape(-1, covariates_blocks.size(-1))
    extended_bags = extended_bags[indices_1].reshape(-1, extended_bags.size(-1))
    bags_blocks = bags_blocks[indices_2]
    targets_blocks = targets_blocks[indices_2]

    # Define stochastic batch iterator
    def batch_iterator(batch_size):
        # Define infinite sampler from HR datataset
        def hr_iterator(batch_size):
            buffer = torch.ones(len(covariates_blocks)).to(device)
            while True:
                idx = buffer.multinomial(batch_size)
                x = covariates_blocks[idx]
                extended_y = extended_bags[idx]
                yield x, extended_y
        # Define iteration loop over LR dataset
        rdm_indices = torch.randperm(len(targets_blocks)).to(device)
        leftovers_sampler = hr_iterator(batch_size=batch_size_cme)
        for idx in rdm_indices.split(batch_size):
            y = bags_blocks[idx]
            z = targets_blocks[idx]
            x, extended_y = next(leftovers_sampler)
            yield x, y, extended_y, z

    # Define variational CME process likelihood
    likelihood = CMPLikelihood(use_individuals_noise=use_individuals_noise)

    # Set model in training mode
    model = model.train().to(device)
    likelihood = likelihood.train().to(device)

    # Define optimizer and elbo module
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    elbo = BagVariationalELBO(likelihood, model, num_data=len(targets_blocks), beta=beta)
    if seed:
        torch.random.manual_seed(seed)

    # Compute unnormalization mean shift and scaling for prediction
    mean_shift = target_field.values.mean()
    std_scale = target_field.values.std()

    # Initialize progress bar
    epoch_bar = Bar("Epoch", max=n_epochs)
    epoch_bar.finish()

    # Logs record
    logs = dict()

    for epoch in range(n_epochs):

        batch_bar = Bar("Batch", max=len(targets_blocks) // batch_size)
        epoch_loss = 0

        for x, y, extended_y, z in batch_iterator(batch_size):
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute q(f)
            q = model(x)

            # Compute tensors needed for ELBO computation
            elbo_kwargs = model.get_elbo_computation_parameters(bags_values=y,
                                                                extended_bags_values=extended_y)

            # Compute negative ELBO loss
            loss = -elbo(variational_dist_f=q,
                         target=z,
                         **elbo_kwargs)

            # Take gradient step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_bar.suffix = f"Running ELBO {-loss.item()}"
            batch_bar.next()

        if epoch % log_every == 0:
            # Compute posterior distribution at current epoch and store logs
            individuals_posterior = predict_downscaling_variational_cmp(model=model,
                                                                        covariates_grid=covariates_grid,
                                                                        mean_shift=mean_shift,
                                                                        std_scale=std_scale)
            epoch_logs = get_epoch_logs(model, likelihood, individuals_posterior, groundtruth_field)
            epoch_logs.update({'loss': epoch_loss / (len(targets_blocks) // batch_size)})
            logs[epoch + 1] = epoch_logs
            with open(os.path.join(dump_dir, 'running_logs.yaml'), 'w') as f:
                yaml.dump({'epoch': logs}, f)

            # Dump plot of posterior prediction at current epoch
            if plot and epoch % plot_every == 0:
                _ = plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field, indices_1)
                plt.savefig(os.path.join(dump_dir, f'png/epoch_{epoch}.png'))
                plt.close()
            epoch_bar.next()
            epoch_bar.finish()

            # Empty cache if using GPU
            if torch.cuda.is_available():
                with torch.cuda.device(f"cuda:{device_idx}"):
                    del individuals_posterior
                    torch.cuda.empty_cache()

    # Save model training state
    state = {'epoch': n_epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(dump_dir, 'state.pt'))


def get_epoch_logs(model, likelihood, individuals_posterior, groundtruth_field):
    # Compute MSE, MAE, MB, Pearson Corr., SSIM
    epoch_logs = compute_metrics(individuals_posterior, groundtruth_field)

    # Record model hyperparameters
    k_spatial_kernel, k_feat_kernel = model.individuals_kernel.kernels
    k_spatial_lengthscales = k_spatial_kernel.base_kernel.lengthscale[0].detach().tolist()
    k_feat_lengthscales = k_feat_kernel.base_kernel.lengthscale[0].detach().tolist()

    l_spatial_kernel, l_feat_kernel = model.bag_kernel.kernels
    l_spatial_lengthscales = l_spatial_kernel.base_kernel.lengthscale[0].detach().tolist()
    l_feat_lengthscales = l_feat_kernel.base_kernel.lengthscale[0].detach().tolist()

    epoch_logs.update({'aggregate_noise': likelihood.noise.detach().item(),
                       'k_spatial_outputscale': k_spatial_kernel.outputscale.detach().item(),
                       'k_lengthscale_lat': k_spatial_lengthscales[0],
                       'k_lengthscale_lon': k_spatial_lengthscales[1],
                       'k_feat_outputscale': k_feat_kernel.outputscale.detach().item(),
                       'k_lengthscale_alt': k_feat_lengthscales[0],
                       'k_lengthscale_albisccp': k_feat_lengthscales[1],
                       'k_lengthscale_clt': k_feat_lengthscales[2],
                       'l_spatial_outputscale': l_spatial_kernel.outputscale.detach().item(),
                       'l_lengthscale_lat': l_spatial_lengthscales[0],
                       'l_lengthscale_lon': l_spatial_lengthscales[1],
                       'l_feat_outputscale': l_feat_kernel.outputscale.detach().item(),
                       'l_lengthscale_pctisccp': l_feat_lengthscales[0]})
    if model.noise_kernel:
        epoch_logs.update({'indiv_noise': model.noise_kernel.outputscale.detach().item()})
    return epoch_logs


@PREDICTERS.register('variational_cmp')
def predict_downscaling_variational_cmp(model, covariates_grid, mean_shift, std_scale, **kwargs):
    # Set model in evaluation mode
    model.eval()

    # Compute standardized posterior distribution on individuals
    with torch.no_grad():
        logging.info("\n Infering deconditioning posterior on HR pixels...")
        individuals_posterior = model(covariates_grid.view(-1, covariates_grid.size(-1)))

    # Rescale by mean and std from observed aggregate target field
    mean_posterior = mean_shift + std_scale * individuals_posterior.mean.cpu()
    lazy_covariance_posterior = (std_scale**2) * individuals_posterior.lazy_covariance_matrix.cpu()
    individuals_posterior = gpytorch.distributions.MultivariateNormal(mean=mean_posterior,
                                                                      covariance_matrix=lazy_covariance_posterior)

    # Set model back to training mode
    model.train()
    return individuals_posterior
