import os
import yaml
import torch
import gpytorch
from progress.bar import Bar
from models import VariationalCMEProcess, CMEProcessLikelihood, MODELS, TRAINERS, PREDICTERS
from core.metrics import compute_metrics


@MODELS.register('variational_cme_process')
def build_swiss_roll_variational_cme_process(individuals, lbda, n_inducing_points, seed, **kwargs):
    """Hard-coded initialization of Variational CME Process module used for swiss roll experiment

    Args:
        individuals (torch.Tensor)
        lbda (float)
        n_inducing_points (int)
        seed (int)

    Returns:
        type: VariationalCMEProcess

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    individuals_mean = gpytorch.means.ZeroMean()

    # Define individuals kernel
    base_individuals_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_individuals_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    individuals_kernel = gpytorch.kernels.ScaleKernel(base_individuals_kernel)

    # Define bags kernels
    base_bag_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_bag_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    bag_kernel = gpytorch.kernels.ScaleKernel(base_bag_kernel)

    # Select inducing points
    if seed:
        torch.random.manual_seed(seed)
    rdm_idx = torch.randperm(len(individuals))[:n_inducing_points]
    inducing_points = individuals[rdm_idx]

    # Define model
    model = VariationalCMEProcess(individuals_mean=individuals_mean,
                                  individuals_kernel=individuals_kernel,
                                  bag_kernel=bag_kernel,
                                  inducing_points=inducing_points,
                                  lbda=lbda)
    return model


@TRAINERS.register('variational_cme_process')
def train_swiss_roll_variational_cme_process(model, individuals, bags_values, aggregate_targets, bags_sizes,
                                             groundtruth_individuals, groundtruth_targets, lr, n_epochs, beta, dump_dir, **kwargs):
    """Hard-coded training script of Variational CME Process for swiss roll experiment

    Args:
        model (VariationalGP)
        individuals (torch.Tensor)
        aggregate_targets (torch.Tensor)
        bags_sizes (list[int])
        lr (float)
        n_epochs (int)
        beta (float)

    """
    # Define variational CME process likelihood
    likelihood = CMEProcessLikelihood()

    # Set model in training mode
    model.train()
    likelihood.train()

    # Define optimizer and elbo module
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(aggregate_targets), beta=beta)

    # Extend bags tensor to match individuals size
    extended_bags_values = torch.cat([x.unsqueeze(0).repeat(bag_size, 1) for (x, bag_size) in zip(bags_values, bags_sizes)])

    # Initialize progress bar
    bar = Bar("Epoch", max=n_epochs)

    # Metrics record
    metrics = dict()

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute q(f)
        q = model(individuals)

        # Compute tensors needed for ELBO computation
        root_inv_extended_bags_covar, bags_to_extended_bags_covar = model.get_elbo_computation_parameters(bags_values=bags_values,
                                                                                                          extended_bags_values=extended_bags_values)

        # Compute negative ELBO loss
        loss = -elbo(variational_dist_f=q,
                     target=aggregate_targets,
                     root_inv_extended_bags_covar=root_inv_extended_bags_covar,
                     bags_to_extended_bags_covar=bags_to_extended_bags_covar)

        # Take gradient step
        loss.backward()
        optimizer.step()

        bar.suffix = f"ELBO {-loss.item()}"
        bar.next()

        # Compute posterior distribution at current epoch and store metrics
        individuals_posterior = predict_swiss_roll_variational_cme_process(model=model,
                                                                           individuals=groundtruth_individuals)
        epoch_metrics = compute_metrics(individuals_posterior, groundtruth_targets)
        metrics[epoch + 1] = epoch_metrics
        with open(os.path.join(dump_dir, 'running_metrics.yaml'), 'w') as f:
            yaml.dump({'epoch': metrics}, f)


@PREDICTERS.register('variational_cme_process')
def predict_swiss_roll_variational_cme_process(model, individuals, **kwargs):
    """Hard-coded prediciton of individuals posterior for Variational CME Process on
    swiss roll experiment

    Args:
        model (VariationalCMEProcess)
        individuals (torch.Tensor)

    Returns:
        type: gpytorch.distributions.MultivariateNormal

    """
    # Set model in evaluation mode
    model.eval()

    # Compute predictive posterior on individuals
    with torch.no_grad():
        individuals_posterior = model(individuals)

    # Set back to trainind mode
    model.train()
    return individuals_posterior
