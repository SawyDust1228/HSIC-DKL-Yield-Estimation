from statistics import mean
import torch
import gpytorch
import botorch
from tqdm import tqdm
from botorch.distributions import Kumaraswamy
from torch import dropout, nn
from torch.utils.data import TensorDataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, data_dim_out = 2):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 1000))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear_final', torch.nn.Linear(1000, data_dim_out))

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, data_dim, data_dim_out):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        # self.c1 = nn.Parameter(torch.rand(data_dim, dtype=torch.float32) * 3 + 0.1)  
        # self.c0 = nn.Parameter(torch.rand(data_dim, dtype=torch.float32) * 3 + 0.1)
        self.feature_extractor = LargeFeatureExtractor(data_dim=data_dim, data_dim_out=data_dim_out)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(ard_num_dims=data_dim_out) + gpytorch.kernels.MaternKernel(ard_num_dims=data_dim_out))
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(lower_bound=-1., upper_bound=1.)

    def forward(self, x):
        # k = Kumaraswamy(concentration1=self.c1, concentration0=self.c0)
        # x = k.icdf(x)
        x = self.feature_extractor(x)
        x = self.scale_to_bounds(x)  # Make the NN values "nice"
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class MyGP():
    def __init__(self, train_x, train_y, data_dim, data_dim_out = 2, training_iterations=50):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.data_dim = data_dim
        self.model = GPRegressionModel(self.train_x, self.train_y, self.likelihood, data_dim, data_dim_out)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.training_iterations = training_iterations
        self.state_dict = None
    
    def renew(self, train_x, train_y):
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)

    def train(self):
        self.model.train()
        self.likelihood.train()
        if self.state_dict is not None:
            self.model.load_state_dict(self.state_dict)

        iterator = tqdm(range(self.training_iterations))
        for i in iterator:
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            iterator.set_postfix()
            self.optimizer.step()
        self.save_dict()

    def test(self, test_x, batch_size = 50):    
        self.model.eval()
        self.likelihood.eval()
        result = torch.tensor(0.)
        var_result = torch.tensor(0.)
        length = test_x.shape[0] // batch_size
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            for i in range(batch_size):
                test_x_cuda = test_x[i * length: length * (i + 1), :].to(device)
                preds = self.likelihood(self.model(test_x_cuda))
                mean, var = preds.mean, preds.variance
                mean = mean.cpu()
                result = torch.hstack([result, mean])
                var = var.cpu()
                var_result = torch.hstack([var_result, var])
        return result[1:], var_result[1:]

    def get_model(self):
        return self.model

    def save_dict(self):
        self.state_dict = self.model.state_dict

    
