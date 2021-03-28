"""
Description : Runs swiss roll experiment

    (1) - Generates bagged swiss roll dataset
    (2) - Fits aggregate model hyperparameters on generated dataset
    (3) - Compute model prediction on indiviuals

Usage: run_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir> [--nbags=<nbags>] [--mean_bag_size=<mean_bag_size>] [--std_bag_size=<std_bag_size>] [--seed=<seed>]

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --nbags=<nbags>                  Number of bags to generate.
  --mean_bag_size=<mean_bag_size>  Mean size of sampled bags.
  --std_bag_size=<std_bag_size>    Size standard deviation of sampled bags.
  --o=<output_dir>                 Output directory.
  --plot                           Outputs scatter plots.
  --seed=<seed>                    Random seed.
"""
import os
import yaml
import logging
from docopt import docopt
import torch
import matplotlib.pyplot as plt
import core.generation as gen
import core.visualization as vis
from models import build_model, train_model, predict


def main(args, cfg):
    # Generate bagged swiss roll dataset
    bags_sizes, individuals, bags_values, aggregate_targets, X_gt, t_gt = make_dataset(cfg['dataset'])
    logging.info("Generated bag swiss roll dataset\n")

    # Save dataset scatter plot
    if args['--plot']:
        dump_plot(plotting_function=vis.plot_dataset,
                  filename='dataset.png',
                  output_dir=args['--o'],
                  individuals=individuals,
                  groundtruth_individuals=X_gt,
                  targets=t_gt,
                  aggregate_targets=aggregate_targets,
                  bags_sizes=bags_sizes)

    # Create model
    cfg['model'].update(individuals=individuals,
                        bags_values=bags_values,
                        aggregate_targets=aggregate_targets,
                        bags_sizes=bags_sizes)
    model = build_model(cfg['model'])
    logging.info(f"Initialized model \n{model}\n")

    # Fit hyperparameters
    logging.info("Fitting model hyperparameters\n")
    cfg['training'].update(model=model,
                           individuals=individuals,
                           bags_values=bags_values,
                           aggregate_targets=aggregate_targets,
                           bags_sizes=bags_sizes)
    train_model(cfg['training'])

    # Compute individuals predictive posterior and plot prediction
    logging.info("Predicting individuals posterior\n")
    predict_kwargs = {'name': cfg['model']['name'],
                      'model': model,
                      'individuals': X_gt}
    individuals_posterior = predict(predict_kwargs)

    # Save prediction scatter plot
    if args['--plot']:
        dump_plot(plotting_function=vis.plot_grountruth_prediction,
                  filename='prediction.png',
                  output_dir=args['--o'],
                  individuals_posterior=individuals_posterior,
                  groundtruth_individuals=X_gt,
                  targets=t_gt)

    # Evaluate mean metrics
    logging.info("Evaluating model\n")
    evaluate_model(cfg=cfg,
                   model=model,
                   individuals_posterior=individuals_posterior,
                   X_gt=X_gt,
                   t_gt=t_gt,
                   output_dir=args['--o'])


def make_dataset(cfg):
    """Generates bagged swiss-roll dataset

    Args:
        cfg (dict): input arguments

    Returns:
        type: list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor

    """
    # Sample bags sizes
    bags_sizes = gen.sample_bags_sizes(mean_bag_size=cfg['mean_bag_size'],
                                       std_bag_size=cfg['std_bag_size'],
                                       n_bags=cfg['nbags'],
                                       seed=cfg['seed'])
    n_samples = sum(bags_sizes)

    # Generate groundtruth and uniform swiss rolls
    X_gt, t_gt = gen.make_swiss_roll(n_samples=cfg['n_samples_groundtruth'],
                                     groundtruth=True,
                                     standardize=True,
                                     seed=cfg['seed'])
    individuals, _ = gen.make_swiss_roll(n_samples=n_samples,
                                         groundtruth=False,
                                         standardize=True,
                                         seed=cfg['seed'])

    # Aggregate individuals into bags
    bags_values, bags_heights = gen.aggregate_bags(X=individuals, bags_sizes=bags_sizes)

    # Compute bags aggregate target based on groundtruth
    aggregate_targets = gen.aggregate_targets(X=X_gt, t=t_gt, bags_heights=bags_heights)

    return bags_sizes, individuals, bags_values, aggregate_targets, X_gt, t_gt


def dump_plot(plotting_function, filename, output_dir, *plot_args, **plot_kwargs):
    """Plot dumping utility

    Args:
        plotting_function (callable): plotting utility
        filename (str): name of saved png file
        output_dir (str): directory where file is dumped

    """
    _ = plotting_function(*plot_args, **plot_kwargs)
    dump_path = os.path.join(output_dir, filename)
    plt.savefig(dump_path)
    plt.close()
    logging.info(f"Plot saved at {dump_path}\n")


def evaluate_model(cfg, model, individuals_posterior, X_gt, t_gt, output_dir):
    """Computes average NLL and MSE on individuals and dumps into YAML file
    """
    # Compute mean square error on individuals posterior
    mse = torch.pow(individuals_posterior.mean - t_gt, 2).mean()

    # Select subset of individuals for NLL computation - scalability
    torch.random.manual_seed(cfg['evaluation']['seed'])
    rdm_idx = torch.randperm(X_gt.size(0))
    n_individuals = cfg['evaluation']['n_samples_nll']
    sub_individuals = X_gt[rdm_idx][:n_individuals]
    sub_individuals_target = t_gt[rdm_idx][:n_individuals]

    # Compute model NLL on subset
    predict_kwargs = {'name': cfg['model']['name'],
                      'model': model,
                      'individuals': sub_individuals}
    sub_individuals_posterior = predict(predict_kwargs)
    with torch.no_grad():
        nll = -sub_individuals_posterior.log_prob(sub_individuals_target).div(n_individuals)

    # Record and dump as YAML file
    individuals_metrics = {'mse': mse.item(), 'nll': nll.item()}
    dump_path = os.path.join(output_dir, 'metrics.yaml')
    with open(dump_path, 'w') as f:
        yaml.dump(individuals_metrics, f)
    logging.info(f"Metrics : {individuals_metrics}\n")


def update_cfg(cfg, args):
    """Updates loaded configuration file with specified command line arguments

    Args:
        cfg (dict): loaded configuration file
        args (dict): script execution arguments

    Returns:
        type: dict

    """
    if args['--nbags']:
        cfg['dataset']['nbags'] = int(args['--nbags'])
    if args['--mean_bag_size']:
        cfg['dataset']['mean_bag_size'] = int(args['--mean_bag_size'])
    if args['--std_bag_size']:
        cfg['dataset']['std_bag_size'] = int(args['--std_bag_size'])
    if args['--seed']:
        cfg['dataset']['seed'] = int(args['--seed'])
    return cfg


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)
    cfg = update_cfg(cfg, args)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)

    # Run session
    main(args, cfg)