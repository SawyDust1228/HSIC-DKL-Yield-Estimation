##
# @author Shuo Yin
# @date   July 2022
# @brief  Feature Selection
# @email  yinshuo991229@gmail.com
##

from pyHSICLasso import HSICLasso
import torch
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
class FeatureSelection:

    def __init__(self) -> None:
        self.hsic_lasso = HSICLasso()
        self.dim = -1

    def fit(self, X, Y, num = 10):
        self.hsic_lasso.input(X, Y)
        self.hsic_lasso.regression(num)
        self.dim = num
        mask = []
        for i in range(0, X.shape[1]):
            if i in self.hsic_lasso.get_index():
                mask.append(True)
            else:
                mask.append(False)
        self.mask = torch.tensor(mask)

    def transform(self, X):
        return X[:, self.mask]

    def get_mask(self):
        return self.mask

    def get_dim(self):
        if self.dim == -1:
            raise "Not Run The Feature selection yet"
        return self.dim

class PCAFeatureSelector:
    def __init__(self, n_components, auto = False) -> None:
        self.n_components = n_components
        if not auto:
            self.pca = PCA(n_components=n_components, svd_solver="full")
        else:
            self.pca = PCA(n_components="mle", svd_solver="full")

    def fit(self, x):
        self.pca.fit(X=x)

    def transform(self, x):
        return self.pca.transform(X=x)
        

class KernelPCAFeatureSelector:
    def __init__(self, n_components) -> None:
        self.kpca = KernelPCA(n_components=n_components, kernel="rbf",n_jobs=4)

    def fit(self, x):
        self.kpca.fit(x)

    def transform(self, x):
        return self.kpca.transform(x)

