import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.special import digamma
from sklearn.gaussian_process import GaussianProcessRegressor
import gpytorch
from tqdm import tqdm
from src.pes.EP import *
import time
from PIL import Image

import sys
# sys.path.append("../deconditional-downscaling/")
sys.path.append("experiments/swiss_roll/")
from models import build_model, train_model, predict 
import yaml 
import torch
import wandb

import numpy as np
import torch
from DMES import qDeMaxValueEntropy
from botorch.optim import optimize_acqf
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
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds, normalisation = False) -> None:
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
        self.random_seeds = random_seeds
        self.normalisation = normalisation
        
        os.makedirs(dump_dir, exist_ok=True)
        
    def update_model(self):
        if self.normalisation:
            self.muz, self.sigmaz = self.z_rewards.mean(), self.z_rewards.std()
            normalised_z_rewards = (self.z_rewards - self.muz) / self.sigmaz
        else:
            self.muz, self.sigmaz = 0, 1
            normalised_z_rewards = self.z_rewards
        
        self.cfg['model'].update(individuals=self.xs,
                                extended_bags_values=self.ys,
                                bags_values=self.y_recs,
                                aggregate_targets=normalised_z_rewards,
                                bags_sizes=len(self.ys))
        self.model = build_model(self.cfg['model'])
        
        self.cfg['training'].update(model=self.model,
                            individuals=self.xs,
                            extended_bags_values=self.ys,
                            bags_values=self.y_recs,
                            aggregate_targets=normalised_z_rewards,
                            bags_sizes=len(self.y_recs),
                            groundtruth_individuals=self.x_space,
                            groundtruth_bags_sizes=len(self.x_space),
                            groundtruth_targets=self.f_oracle(self.x_space),
                            chunk_size=self.cfg['evaluation']['chunk_size_nll'],
                            device_idx='cpu',
                            dump_dir=self.dump_dir
                            )
        train_model(self.cfg['training'])
        
    def predict_fx(self, num_exp=0, num_iter=0, plot_flag = False):
        predict_kwargs = {'name': self.cfg['model']['name'],
                'model': self.model.eval().cpu(),
                'individuals': self.x_space,
                }
        individuals_posterior = predict(predict_kwargs)
        individuals_posterior_mean = self.sigmaz * individuals_posterior.mean + self.muz
        with torch.no_grad():
            stddev = self.sigmaz * individuals_posterior.stddev
            lower_bound = individuals_posterior_mean - 2 * stddev
            upper_bound = individuals_posterior_mean + 2 * stddev
        if plot_flag:
            plt.plot(self.x_space, individuals_posterior_mean, label = 'posterior mean', color='C0')
            plt.fill_between(self.x_space, lower_bound, upper_bound, alpha=0.3, color='C0')
            # plt.plot(groundtruth_individuals, (conf[1]-conf[0]).detach().numpy(), label = '2 * posterior std')
            plt.plot(self.x_space, self.f_oracle(self.x_space), label = 'f(x)', color='C1')
            plt.xlim(self.x_space[0], self.x_space[-1])
            # plt.ylim(1.5 * self.y_space[0], 1.5 * self.y_space[-1])
            # plt.plot(xs, f(xs), '.', label = 'data points')
            plt.plot(self.y_recs, self.z_rewards, '.', label = 'observations z')
            plt.title('Iteration: ' + str(num_iter))
            plt.legend()
            plt.savefig(f"{self.dump_dir}predf/frame_{num_exp}_{num_iter}.png")
            plt.clf()
        return individuals_posterior_mean, stddev, individuals_posterior
    
    def predict_gy(self, num_exp=0, num_iter=0, plot_flag = False):
        # TODO: not in used yet
        md,_, xs_posterior = self.predict_fx(num_exp, num_iter, plot_flag)
        
        with torch.no_grad():
            Kd = self.sigmaz**2 * xs_posterior.covariance_matrix

            N = len(self.y_space)
            λ = 0.01
            Lλ = self.model.bag_kernel(self.y_space, self.y_space).add_diag(λ * N * torch.ones(N))
            chol = torch.linalg.cholesky(Lλ.evaluate())
            
            Lys_ys2 = self.model.bag_kernel(self.y_space, self.x_space).evaluate()
            Lλinv_Lys_ys2 = torch.cholesky_solve(Lys_ys2, chol)
            
            g_cov = Lλinv_Lys_ys2.T @ Kd @ Lλinv_Lys_ys2 # this is the term in (12)
            
            phi_x = self.model.individuals_kernel(self.x_space, self.x_space)
            g_mu =  Lλinv_Lys_ys2.T @  phi_x @ md
            
            return g_mu, g_cov
    
    def init_y_recs(self, j):
        random.seed(int(self.random_seeds[j]))
        # y_recs = np.random.uniform(self.y_space[0], self.y_space[-1], self.init_y_recs_size)
        y_recs = np.array(random.sample(list(self.y_space), self.init_y_recs_size))
        # print("y_recs: ", y_recs.shape)
        z_rewards = np.array([self.g_oracle(y) for y in y_recs]).reshape(-1,)
        print(y_recs)
        return torch.from_numpy(y_recs).float(), torch.from_numpy(z_rewards).float()
        
        
    def rec_policy(self, mean, std, posterior):
        pass
       
    def simulation(self):
        pos_bests = []
        z_rewards_all = []
        plot_flag = False
        if plot_flag:
            os.makedirs(self.dump_dir + 'predf/', exist_ok=True)
            os.makedirs(self.dump_dir + 'obj/', exist_ok=True)
        for j in tqdm(range(self.n_repeat)): 
            # wandb.init(
            #         # set the wandb project where this run will be logged
            #         project="bodd",
                    
            #         # track hyperparameters and run metadata
            #         config={
            #         "policy": self.__class__.__name__,
            #         "f": self.f_oracle,
            #         "g": self.g_oracle,
            #         'num_exp': j
            #         }
            #     )
            pos_best = []
            self.y_recs, self.z_rewards = self.init_y_recs(j)
            for i in tqdm(range(self.n-self.init_y_recs_size)):
                # start a new wandb run to track this script
                
                start_time = time.time()
                self.update_model()
                mean, std, posterior = self.predict_fx(j, i, plot_flag=plot_flag)
                pos_best_j = self.f_oracle(self.x_space[np.argmax(mean)])
                pos_best.append(pos_best_j)
                y_rec, x_opt= self.rec_policy(mean, std, posterior, j, i, plot_flag=plot_flag)
                # self.y_recs = np.append(self.y_recs, y_rec)
                # print("y_rec: ", y_rec.shape)
                # print("self.y_recs: ", self.y_recs.shape)
                # self.y_recs = torch.cat((self.y_recs, y_rec.view(1).float()), dim=0)
                self.y_recs = torch.cat((self.y_recs, y_rec.view(1).float()), dim=0)
                z_reward = torch.tensor(self.g_oracle(y_rec.T))
                # self.z_rewards = np.append(self.z_rewards, z_reward)
                self.z_rewards = torch.cat((self.z_rewards, z_reward.view(1).float()), dim=0)
                # print(len(self.y_recs))
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                opt = torch.min(self.f_oracle(self.x_space))
                regret = torch.abs(opt - pos_best_j)
                
                # wandb.log({
                #     "regret": regret,
                #     "pos_best": pos_best_j,
                #     "x_opt": x_opt,
                #     "y_rec": y_rec,
                #     "z_reward": z_reward,
                #     "elapsed_time": elapsed_time 
                #     })

                print(f"Iteration {i + 1}: Elapsed Time: {elapsed_time} seconds")
            pos_bests.append(pos_best)
            z_rewards_all.append(self.z_rewards)
            if plot_flag:
                # Create a GIF from the frames
                frames = []
                for i in range(self.n-self.init_y_recs_size):
                    frame = Image.open(f"{self.dump_dir}predf/frame_{j}_{i}.png")
                    frames.append(frame)

                # Save the GIF
                frames[0].save(self.dump_dir + "predf/predictions_animation_" + str(j) + '_' + str(i) + '.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)
                
                frames = []
                for i in range(self.n-self.init_y_recs_size):
                    frame = Image.open(f"{self.dump_dir}obj/frame_{j}_{i}.png")
                    frames.append(frame)

                # Save the GIF
                frames[0].save(self.dump_dir + "obj/obj_animation_" + str(j) + '_' + str(i) + '.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)
            # wandb.finish()
        # self.evalaution(pos_bests)
        return pos_bests, z_rewards_all
    
class BayesOpt_Random(BayesOpt):
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds) -> None:
        super().__init__(dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds)
        
    def rec_policy(self, mean, std, posterior, num_exp, num_iter, plot_flag):
        # np.random.seed(self.random_seed)
        # rec_y = np.random.uniform(self.y_space[0], self.y_space[-1], 1)
        rec_y = np.array(random.sample(list(self.y_space), 1))
        if plot_flag:
            plt.axvline(x=rec_y, color='red', linestyle='--', label='rec')
            plt.xlabel('y')
            plt.title('Iteration: ' + str(num_iter))
            plt.legend()
            plt.savefig(f"{self.dump_dir}obj/frame_{num_exp}_{num_iter}.png")
            plt.clf()
        return torch.tensor(rec_y), None
    
class MES(BayesOpt):
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds, maximize = True) -> None:
        super().__init__(dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds)
        self.maximize = maximize
        
    def rec_policy(self, mean, std, posterior, num_exp, num_iter, plot_flag=False):
        tkwargs = {"dtype": torch.float, "device": "cpu"}
        # bounds = torch.tensor([[self.y_space[0]], [self.y_space[-1]]], **tkwargs)
        # candidate_set = torch.rand(
        #     1000, bounds.size(1), device=bounds.device, dtype=bounds.dtype
        # )
        # candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        
        x_bounds = torch.tensor([[self.x_space[0]], [self.x_space[-1]]], **tkwargs)
        # x_set = torch.rand(
        #     1000, x_bounds.size(1), device=x_bounds.device, dtype=x_bounds.dtype
        # )
        x_set=torch.linspace(0,1,steps=1000,device=x_bounds.device, dtype=x_bounds.dtype)
        x_set = x_bounds[0] + (x_bounds[1] - x_bounds[0]) * x_set
 
        qMES = qDeMaxValueEntropy(self.model,x_set=x_set, maximize=self.maximize, normalise_para = [self.muz, self.sigmaz])
        fwd_y = self.y_space #.unsqueeze(-1) #.unsqueeze(-1)
        mes_y = qMES(fwd_y).detach().numpy()
        # print("mes_y: ", mes_y.shape)
        # print('mes_y.argmax(): ', mes_y.argmax())
        # rec = torch.tensor([self.y_space[mes_y.argmax()]]) # TODO: check max or min 
        rec = self.y_space[mes_y.argmax()].reshape(1,-1)
        
        # report posterior of g
        fig = plt.figure()
        
        with torch.no_grad():
            posterior_m = self.model(fwd_y)
            mean = self.sigmaz * posterior_m.mean + self.muz
            std = self.sigmaz * posterior_m.stddev
            
        # This is the posterior of g -- plot
        # plt.plot(fwd_y, mean, label = 'posterior mean', color='C0')
        # plt.fill_between(fwd_y, mean - 2 * std, mean + 2 * std, alpha=0.3, color='C0')
        # plt.plot(self.y_recs[:-2], self.z_rewards[:-2], '.', label = 'observations')
        # plt.plot(self.y_recs[-1], self.z_rewards[-1], '.', color='red')
        # # plt.plot(groundtruth_individuals, (conf[1]-conf[0]).detach().numpy(), label = '2 * posterior std')
        # # plt.plot(fwd_y, self.g_oracle(fwd_y), label = 'g(y)', color='C1')
        # plt.title('posterior of g and observations')
        # plt.xlim(fwd_y[0], fwd_y[-1])
        # plt.ylim(-5, 5)
        # # plt.ylim(0,1)
        # plt.legend()
        # # plt.savefig(f"{self.dump_dir}predf/frame_{num_exp}_{num_iter}.png")
        # plt.show()
        # plt.close(fig)
        
        if plot_flag:
            plt.plot(self.y_space, mes_y, label='objective')
            plt.xlim(self.y_space[0], self.y_space[-1])
            # plt.ylim(-3, 1) 
            plt.axvline(x=rec, color='red', linestyle='--', label='rec')
            plt.xlabel('y')
            plt.title('Iteration: ' + str(num_iter))
            plt.legend()
            plt.savefig(f"{self.dump_dir}obj/frame_{num_exp}_{num_iter}.png")
            plt.clf()
        return torch.tensor(rec), None
        
   
class PES(BayesOpt):
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, y_min, λ, num_opt_samples, random_seeds) -> None:
        super().__init__(dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds)
        self.y_min = y_min
        self.λ = λ
        self.num_opt_samples = num_opt_samples
        
    def sample_opt(self, individual_posterior):
        pos_sample = individual_posterior.sample()
        x_opt = self.x_space[np.argmin(pos_sample)]
        x_opt = torch.from_numpy(np.array([x_opt])).float()
        return x_opt
    
    def conditional_expectation(self, K):
        with torch.no_grad():
            N = len(self.ys)
            Lλ = self.model.bag_kernel(self.ys, self.ys).add_diag(self.λ * N * torch.ones(N))
            chol = torch.linalg.cholesky(Lλ.evaluate())
            
            Lys_ys2 = self.model.bag_kernel(self.ys, self.y_space).evaluate()
            Lλinv_Lys_ys2 = torch.cholesky_solve(Lys_ys2, chol)
            
            B = Lλinv_Lys_ys2.T @ K @ Lλinv_Lys_ys2 
            
            return torch.diag(B)
    
    def PES_term1(self, noise):
        predict_kwargs = {'name': self.cfg['model']['name'],
                  'model': self.model.eval().cpu(),
                  'individuals': self.xs}
        xs_posterior = predict(predict_kwargs)
        
        with torch.no_grad():
            Kd = self.sigmaz**2 * xs_posterior.covariance_matrix
            B = self.conditional_expectation(Kd) # this is the term in (13)
            entropy_z_given_y = 0.5 * np.log(2 * np.pi * np.e * (B + noise ** 2))
            # print('entropy: ', entropy_z_given_y)
            # print('noise: ', noise)
        return entropy_z_given_y
    
    def PES_term2(self, posterior, l, sigma, noise):
        hess_at_min = np.zeros((1,1))
        entropy_list = []
        x_opt_list = []
        while len(entropy_list)<self.num_opt_samples:
            # try:
            x_opt = self.sample_opt(posterior)
            x_opt_list.append(x_opt)
            # print('x_opt ', x_opt)
            K, K_star_min, K_plus_W_tilde_inverse, m_f_minimum, v_f_minimum, c_and_m = Expectation_Propagation(nObservations=[self.xs.reshape(-1,1), self.ys.reshape(-1,1), self.y_recs.reshape(-1,1)], value_of_nObservations=self.z_rewards, num_of_obser=len(self.y_recs), x_minimum=x_opt.reshape(-1,1), d=1, l_vec=l, sigma=sigma, noise=noise, hess_at_min=hess_at_min, model=self.model, λ=self.λ, y_min = self.y_min)
            #K_star is the cross-covariance column evaluated between f(x) and [c;z], its dimension is 1x(n+d+d*(d-1)/2 +d+1)
            #with f(x_min) being the last element
            K_star = compute_cov_xPrime_cz(self.xs.reshape(-1,1), [self.xs, self.ys, self.y_recs], x_opt, len(self.y_recs), self.model, noise, l)
            #m_f_evaluated is the other element of the m_f vector
            m_f_evaluated = np.dot(np.dot(K_star, K_plus_W_tilde_inverse), c_and_m)
            #v_f_evaluated is one of the element of the V_f vector, which corresponds to the variance of the 
            #posterior distribution of f(x)
            v_f_evaluated = sigma - np.dot(np.dot(K_star, K_plus_W_tilde_inverse), K_star.T)
            
            B = self.conditional_expectation(torch.from_numpy(v_f_evaluated).float())
            
            entropy = 0.5*np.log(2*np.pi*np.exp(1)*(B + noise ** 2))
            # entropy = 0.5*np.log(2*np.pi*np.exp(1)*(B))
            entropy_list.append(entropy)
            
            # except:
            #     pass
        
        # Calculate the mean along the specified dimension (0 in this case)
        x_opt = torch.mean(torch.stack(x_opt_list, dim=0), dim=0)
        return torch.mean(torch.stack(entropy_list, dim =0), dim = 0), x_opt
            
            
        
    def rec_policy(self, mean, std, posterior, num_exp, num_iter, plot_flag = True):
        with torch.no_grad():
            l = self.model.individuals_kernel.base_kernel.lengthscale.numpy()
            sigma = self.model.individuals_kernel.outputscale.numpy()
            noise = self.model.likelihood.noise.numpy()
            # print('noise ** 2: ', noise ** 2)
            
        term1 = self.PES_term1(noise)
        term2, x_opt = self.PES_term2(posterior, l, sigma, noise)
        objective = term1 - term2
        rec_y = self.y_space[np.argmax(objective)]
        # print(objective)
        if plot_flag:
            plt.plot(self.y_space, objective, label='objective')
            plt.xlim(self.y_space[0], self.y_space[-1])
            # plt.ylim(-3, 1) # TODO: change
            plt.axvline(x=rec_y, color='red', linestyle='--', label='rec')
            plt.xlabel('y')
            plt.title('Iteration: ' + str(num_iter))
            plt.legend()
            plt.savefig(f"{self.dump_dir}obj/frame_{num_exp}_{num_iter}.png")
            plt.clf()
        return rec_y, x_opt
        

class BALD(BayesOpt):
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds) -> None:
        super().__init__(dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds)
        
    def rec_policy(self):
        pass

class BayesOpt_UCB(BayesOpt):
    def __init__(self, dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds, alpha = 2) -> None:
        super().__init__(dataset1, init_y_recs_size, y_space, x_space, f_oracle, g_oracle, num_round, num_repeat, cdf_dir, dump_dir, random_seeds)
        self.x_to_y_model, self.x_to_y_likelihood = self.build_transform_x_to_y()
        self.alpha = alpha
        
        
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
    
    def rec_policy(self, mean, std, posterior, j, i, plot_flag=False):
        rec_x = self.x_space[np.argmin(mean - self.alpha * std)]
        rec_y = self.transform_x_to_y(rec_x)
        # print(rec_y)
        return torch.tensor(rec_y).reshape(1,-1), None
    
    
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