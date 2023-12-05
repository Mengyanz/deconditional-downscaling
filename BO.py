import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import digamma
from sklearn.gaussian_process import GaussianProcessRegressor
import gpytorch
from tqdm import tqdm

import sys
# sys.path.append("../deconditional-downscaling/")
sys.path.append("experiments/swiss_roll/")
from models import build_model, train_model, predict 
import yaml 
import torch

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class BayesOpt():
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seed=24) -> None:
        self.xs, self.ys= dataset1 # dataset 1 - x, y pairs
        self.y_space = y_space
        self.x_space = x_space
        self.n = num_round
        self.n_repeat = num_repeat
        self.f_oracle = f_oracle
        self.g_oracle = g_oracle
        with open(cdf_dir, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.dump_dir = dump_dir
        self.init_y_recs_size = init_y_recs_size
        self.random_seed = random_seed
        
    def update_model(self):
        self.muz, self.sigmaz = self.z_rewards.mean(), self.z_rewards.std()
        normalised_z_rewards = (self.z_rewards - self.muz) / self.sigmaz
        
        self.cfg['model'].update(individuals=torch.tensor(self.xs),
                                extended_bags_values=torch.tensor(self.ys),
                                bags_values=torch.tensor(self.y_recs),
                                aggregate_targets=torch.tensor(normalised_z_rewards),
                                bags_sizes=len(self.ys))
        self.model = build_model(self.cfg['model'])
        
        self.cfg['training'].update(model=self.model,
                            individuals=torch.tensor(self.xs),
                            extended_bags_values=torch.tensor(self.ys),
                            bags_values=torch.tensor(self.y_recs),
                            aggregate_targets=torch.tensor(normalised_z_rewards),
                            bags_sizes=len(self.ys),
                            groundtruth_individuals=torch.tensor(self.x_space),
                            groundtruth_bags_sizes=len(self.x_space),
                            groundtruth_targets=torch.tensor(self.f_oracle(self.x_space)),
                            chunk_size=self.cfg['evaluation']['chunk_size_nll'],
                            device_idx='cpu',
                            dump_dir=self.dump_dir
                            )
        train_model(self.cfg['training'])
        
    def predict_fx(self):
        predict_kwargs = {'name': self.cfg['model']['name'],
                'model': self.model.eval().cpu(),
                'individuals': torch.tensor(self.x_space),
                'bags_sizes': torch.tensor(self.f_oracle(self.x_space))} # todo: CHECK
        individuals_posterior = predict(predict_kwargs)
        individuals_posterior_mean = self.sigmaz * individuals_posterior.mean + self.muz
        with torch.no_grad():
            stddev = self.sigmaz * individuals_posterior.stddev
            lower_bound = individuals_posterior_mean - 2 * stddev
            upper_bound = individuals_posterior_mean + 2 * stddev
        return individuals_posterior_mean, stddev
    
    def init_y_recs(self):
        # np.random.seed(self.random_seed)
        y_recs = np.random.uniform(self.y_space[0], self.y_space[-1], self.init_y_recs_size)
        z_rewards = self.g_oracle(y_recs)
        return y_recs, z_rewards
        
        
    def rec_policy(self):
        pass
       
    def simulation(self):
        pos_bests = []
        for _ in tqdm(range(self.n_repeat)): 
            pos_best = []
            self.y_recs, self.z_rewards = self.init_y_recs()
            for _ in tqdm(range(self.n-self.init_y_recs_size)):
                self.update_model()
                mean, std = self.predict_fx()
                pos_best.append(self.f_oracle(self.x_space[np.argmin(mean)]))
                y_rec= self.rec_policy(mean, std)
                self.y_recs = np.append(self.y_recs, y_rec)
                z_reward = self.f_oracle(y_rec)
                self.z_rewards = np.append(self.z_rewards, z_reward)
                # print(len(self.y_recs))
            pos_bests.append(pos_best)
        # self.evalaution(pos_bests)
        return pos_bests
    
class BayesOpt_Random(BayesOpt):
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seed=24) -> None:
        super().__init__(dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seed)
        
    def rec_policy(self, mean, std):
        # np.random.seed(self.random_seed)
        y_recs = np.random.uniform(self.y_space[0], self.y_space[-1], 1)
        return y_recs


class BALD(BayesOpt):
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seed=24) -> None:
        super().__init__(dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seed)
        
    def rec_policy(self):
        pass

class BayesOpt_UCB(BayesOpt):
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seed=24) -> None:
        super().__init__(dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seed)
        self.x_to_y_model, self.x_to_y_likelihood = self.build_transform_x_to_y()
        
        
    def build_transform_x_to_y(self):
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(torch.tensor(self.xs), torch.tensor(self.ys), likelihood)
        training_iter = 50

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(torch.tensor(self.xs))
            # Calc loss and backprop gradients
            loss = -mll(output, torch.tensor(self.ys))
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            optimizer.step()
        return model, likelihood
    
    def transform_x_to_y(self, x):      
        # Get into evaluation (predictive posterior) mode
        self.x_to_y_model.eval()
        self.x_to_y_likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.tensor([x])
            observed_pred = self.x_to_y_likelihood(self.x_to_y_model(test_x))
        
        mean= observed_pred.mean.numpy()
        return mean
    
    def rec_policy(self, mean, std, alpha = 2):
        rec_x = self.x_space[np.argmin(mean - alpha * std)]
        rec_y = self.transform_x_to_y(rec_x)
        # print(rec_y)
        return rec_y
    
    
def evalaution(self, pos_bests):
    # plt.plot(range(self.n), self.z_rewards[self.init_y_recs_size:], '.', label = 'z rewards')
    opt = np.min(self.f_oracle(self.x_space))
    regret_mean = np.abs(opt - pos_bests).mean(axis=0)
    regret_std = np.abs(opt - pos_bests).std(axis=0)
    print(regret_mean)
    plt.plot(range(self.n - self.init_y_recs_size), regret_mean)
    plt.fill_between(range(self.n - self.init_y_recs_size), regret_mean - 2 * regret_std, regret_mean + 2 * regret_std, alpha = 0.3)
    plt.xlabel('Round')
    plt.ylabel('Regret')        