
import torch
from torch import device, nn
from models.Model import MyGP, GPRegressionModel
from torch import Tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Entropy(nn.Module):
    def __init__(self, model : GPRegressionModel, threshold = 0.75) -> None:
        super().__init__()
        self.model = model
        self.threshold = torch.tensor(threshold, device=device)

    def getBernuliEntropy(self, p):
        if(p == torch.tensor(0.) or p == torch.tensor(1.)):
            return torch.tensor(0.).to(device=device)

        return -p * torch.log(p) - (1 - p) * torch.log(1 - p)
    
    def get_prob_entropy(self, x):
        pred = self.model.likelihood(self.model(x)) 
        mean = pred.mean
        var = pred.variance
        result = torch.tensor(0.)
        result = result.to(device)
        for i in range(mean.shape[0]):
            normal = torch.distributions.Normal(mean[i], var[i])
            result = torch.hstack([result, torch.tensor(1., device=device) - normal.cdf(self.threshold)])
        return result[1:]

    def forward(self, x):
        center = x.to(device)
        entropy = self.get_prob_entropy(center.reshape(1, -1))
        z = self.getBernuliEntropy(entropy)
        return z

# TODO
class AdaptiveSampling:
    def __init__(self, model : MyGP, mask, train_x, threshold = 1.75, N = 10) -> None:
        self.model = model
        self.mask = mask
        self.train_x = train_x
        self.train_x_mask = train_x[:, mask]
        self.dim = self.train_x_mask.shape[1]
        self.entropyConvolution = Entropy(model.get_model(), threshold=threshold)
        self.N = N

    def get_initial_set(self):
        N = self.N
        y_raw = torch.tensor(0.).to(device)
        for i in range(self.train_x_mask.shape[0]):
            value = self.entropyConvolution(self.train_x_mask[i]).detach().clone()
            y_raw = torch.hstack([y_raw, value.reshape(-1)])
        y_raw = y_raw[1:]
        y_raw = y_raw.cpu()
    
        X, indeces= self.initialize_q_batch(self.train_x_mask, y_raw, N, alpha=0.7)
        # we'll want gradients for the input
        print("initial finished")
        return self.train_x[indeces], indeces, y_raw[indeces]

    def initialize_q_batch(self, X: Tensor, Y: Tensor, n: int, eta = 1.0, alpha = 0.7, beta = 0.05) -> Tensor:

        max_val, max_idx = torch.max(Y, dim=0)
        #print(Y)

        alpha_pos = Y >= alpha * max_val
        while alpha_pos.sum() < n:
            alpha = (1 - beta) * alpha # changed
            alpha_pos = Y >= alpha * max_val
        alpha_pos_idcs = torch.arange(len(Y), device=Y.device)[alpha_pos]
        weights = torch.exp(eta * (Y[alpha_pos] / max_val - 1))
        idcs = alpha_pos_idcs[torch.multinomial(weights, n)]
        if max_idx not in idcs:
            idcs[-1] = max_idx
        # idcs = sort(Y, n)
        return X[idcs], torch.tensor(idcs)
    


