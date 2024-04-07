#%%
import warnings
import torch
import torch.linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

#%%
def pairwise_rbf(X, Y=None, gamma=1.):
    # Add check dtype and device for X and Y
    if Y is None:
        Y = X
    return torch.exp(-gamma*torch.square(torch.cdist(X, Y)))

class Nystroem(TransformerMixin, BaseEstimator):
    def __init__(self, gamma=1., n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit estimator to data.
        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.
        Parameters
        ----------
        X : torch.tensor of dtype=torch.float32 and shape (n_samples, n_features)
            Training data.
        """
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]
        # get basis vectors
        if self.n_components > n_samples:
            n_components = n_samples
            warnings.warn("n_components > n_samples. This is not possible.\n"
                          "n_components was set to n_samples, which results"
                          " in inefficient evaluation of the full kernel.")

        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]

        basis_kernel = pairwise_rbf(basis, gamma=self.gamma)
        # sqrt of kernel matrix on basis vectors
        U, S, V = torch.linalg.svd(basis_kernel) # changed to torch.linalg.svd in newest version of pytorch
        S = torch.maximum(S, torch.tensor(1e-12, device=S.device))
        self.normalization_ = torch.matmul(U / torch.sqrt(S), V) # use V.T instead of V (V is V.T in scipy.linalg,svd)
        self.components_ = basis
        self.component_indices_ = inds
        return self
    
    def transform(self, X):
        """Apply feature map to X.
        Computes an approximate feature map using the kernel
        between some training points and X.
        Parameters
        ----------
        X : torch.tensor of dtype=torch.float32 and shape (n_samples, n_features)
            Training data.
        Returns
        -------
        X_transformed : torch.tensor of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)

        embedded = pairwise_rbf(X, self.components_, gamma=self.gamma)
        return torch.matmul(embedded, self.normalization_.T)

class Scaler(TransformerMixin, BaseEstimator):
    # RobustScaler
    def __init__(self, *, with_centering=True, with_scaling=True,
                quantile_range=(25.0, 75.0), copy=True, unit_variance=False):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the median and quantiles to be used for scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        # at fit, convert sparse matrices to csc for optimized computation of
        # the quantiles
        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        if self.with_centering:
            self.center_, _ = torch.nanmedian(X, dim=0)
        else:
            self.center_ = None

        if self.with_scaling:
            quantiles = torch.quantile(X, torch.tensor([q_min/100., q_max/100.], device=X.device, dtype=X.dtype), dim=0)

            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_[self.scale_==0.] = 1.
            if self.unit_variance:
                adjust = (stats.norm.ppf(q_max / 100.0) -
                          stats.norm.ppf(q_min / 100.0))
                self.scale_ = self.scale_ / adjust
        else:
            self.scale_ = None

        return self

    def transform(self, X):
        """Center and scale the data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the specified axis.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        if self.with_centering:
            Y = X -  self.center_
        if self.with_scaling:
            Y = X / self.scale_
        return Y

    def inverse_transform(self, X):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The rescaled data to be transformed back.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        if self.with_scaling:
            X *= self.scale_
        if self.with_centering:
            X += self.center_
        return X
#%%
def sqrtm(input):
    """
    https://people.cs.umass.edu/~smaji/projects/matrix-sqrt/
    https://github.com/pytorch/pytorch/issues/25481#issuecomment-584822283
    """
    dim = input.shape[0]
    norm = torch.norm(input.double())
    Y = input.double()/norm
    I = torch.eye(dim,dim,device=input.device).double()
    Z = torch.eye(dim,dim,device=input.device).double()
    for i in range(20):
        T = 0.5*(3.0*I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)
    sqrtm = Y*torch.sqrt(norm)
    return sqrtm
#%%
def _check_auto(param):
    return (isinstance(param, str) and (param == 'auto'))

def _to_tensor(X, device):
    if isinstance(X, torch.Tensor):
        if X.device.__str__() == device:
            if X.dtype == torch.float64:
                return X
            else:
                return X.double()
        else:
            return X.double().to(device)
    else:
        return torch.tensor(X, dtype=torch.float64, device=device)
#%%
class _BaseRKHSIV:

    def __init__(self, *args, **kwargs):
        return

    def _get_delta(self, n):
        '''
        delta -> Critical radius
        '''
        delta_scale = 5. if _check_auto(self.delta_scale) else self.delta_scale
        delta_exp = .4 if _check_auto(self.delta_exp) else self.delta_exp
        return delta_scale / (n**(delta_exp))

    def _get_alpha_scale(self):
        return 60. if _check_auto(self.alpha_scale) else self.alpha_scale

    def _get_alpha_scales(self, n):
        return ([c for c in np.geomspace(0.001, 0.05, self.n_alphas)]
                if _check_auto(self.alpha_scales) else self.alpha_scales)

    def _get_alpha(self, delta, alpha_scale):
        return alpha_scale * (delta**4)

    # def _get_kernel(self, X, Y=None):
    #     if callable(self.kernel):
    #         params = self.kernel_params or {}
    #     else:
    #         params = {"gamma": self.gamma,
    #                   "degree": self.degree,
    #                   "coef0": self.coef0}
    #     return pairwise_kernels(X, Y, metric=self.kernel,
    #                             filter_params=True, **params)

    def _get_gamma_f(self, condition):
        if _check_auto(self.gamma_f):
            K_condition_euclidean = torch.square(torch.cdist(condition, condition))
            triuInd = torch.triu_indices(K_condition_euclidean.size(0),K_condition_euclidean.size(0),offset=1)
            gamma_f = 1./(condition.shape[1] * torch.median(K_condition_euclidean[triuInd[0],triuInd[1]]))
            #gamma_f = 1./torch.median(K_condition_euclidean[triuInd[0],triuInd[1]])
            return gamma_f.detach().tolist()
        else:
            return self.gamma_f

    def _get_kernel_f(self, X, Y=None, gamma_f=1.):
        return pairwise_rbf(X, Y, gamma = gamma_f)

    def _get_kernel_h(self, X, Y=None, gamma_h=1.):
        return pairwise_rbf(X, Y, gamma = gamma_h)

#%%
class RKHSIV(_BaseRKHSIV):

    def __init__(self, gamma_h=0.1, gamma_f='auto', 
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto', device='cuda'):
        """
        Parameters:
            gamma_h : the gamma parameter for the rbf kernel of h
            gamma_f : the gamma parameter for the rbf kernel of f
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scale : the scale of the regularization; alpha = alpha_scale * (delta**4)
        """
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h 
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale  # regularization strength from Theorem 5
        self.device = device
        # Check cuda GPU device
        if torch.cuda.is_available():
            self.device = self.device
        else:
            self.device = 'cpu'

    def fit(self, X, y, condition):
        X         = _to_tensor(X, self.device)
        y         = _to_tensor(y, self.device)
        condition = _to_tensor(condition, self.device)
        # Standardize condition and get gamma_f -> Kf -> RootKf
        condition = Scaler().fit_transform(condition)
        self.gamma_f = self._get_gamma_f(condition=condition)
        Kf = self._get_kernel_f(condition, gamma_f=self.gamma_f)
        RootKf = sqrtm(Kf)

        # Standardize X and get Kh
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.X = X.clone()
        Kh = self._get_kernel_h(X, gamma_h=self.gamma_h)

        # delta & alpha
        n = X.shape[0]  # number of samples
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        # M
        # M = RootKf @ torch.linalg.inv(
        #     Kf / (2 * n * delta**2) + torch.eye(n, device=self.device, dtype=X.dtype) / 2) @ RootKf
        M = RootKf @ torch.linalg.lstsq(
            Kf / (2 * n * delta**2) + torch.eye(n, device=self.device, dtype=X.dtype) / 2, RootKf).solution

        #self.a = torch.linalg.pinv(Kh @ M @ Kh + alpha * Kh) @ Kh @ M @ y # torch.pinverse=np.linalg.pinv; torch.linalg.pinv=np.linalg.pinv2
        self.a = torch.linalg.lstsq(Kh @ M @ Kh + alpha * Kh, Kh @ M @ y).solution
        return self

    def predict(self, X):
        X = _to_tensor(X, self.device)
        X = self.transX.transform(X)
        return self._get_kernel_h(X, Y=self.X, gamma_h=self.gamma_h) @ self.a

    def score(self, X, y, M):
        X = _to_tensor(X, self.device)
        y = _to_tensor(y, self.device)
        M = _to_tensor(M, self.device)
        n = X.shape[0]
        #delta = self._get_delta(n)
        #Kf = self._get_kernel_f(Z, gamma_f=self.gamma_f)
        #RootKf = scipy.linalg.sqrtm(Kf).astype(float)
        #M = RootKf @ np.linalg.inv(
        #    Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        y_pred = self.predict(X)
        return ((y - y_pred).T @ M @ (y - y_pred)).data[0] / n**2

class RKHSIVCV(RKHSIV):

    def __init__(self, gamma_f='auto', gamma_hs='auto', n_gamma_hs=20,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=5, device='cuda'):
        """
        Parameters:
            gamma_f : the gamma parameter for the kernel of f
            gamma_hs : the list of gamma parameters for kernel of h
            n_gamma_hs : how many gamma_hs to try
            delta_scale : the scale of the critical radius; delta_n = delta_scale / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scale / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**4)
            n_alphas : how many alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale, gamma_h
        """

        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs=n_gamma_hs
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  # regularization strength from Theorem 5
        self.n_alphas = n_alphas
        self.cv = cv
        self.device = device
        # Check cuda GPU device
        if torch.cuda.is_available():
            self.device = self.device
        else:
            self.device = 'cpu'

    def _get_gamma_hs(self,X):
        if _check_auto(self.gamma_hs):
            K_X_euclidean = torch.square(torch.cdist(X, X))
            triuInd = torch.triu_indices(K_X_euclidean.size(0),K_X_euclidean.size(0),offset=1)
            return 1./torch.quantile(K_X_euclidean[triuInd[0],triuInd[1]], torch.linspace(0.1,0.9,steps=self.n_gamma_hs,device=X.device, dtype=X.dtype))/X.shape[1]
        else:
            return _to_tensor(self.gamma_hs, self.device)

    def fit(self, X, y, condition):
        X         = _to_tensor(X, self.device)
        y         = _to_tensor(y, self.device)
        condition = _to_tensor(condition, self.device)
        # Standardize condition and get gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition = condition)
        Kf = self._get_kernel_f(condition, gamma_f=gamma_f)
        RootKf = sqrtm(Kf)

        # Standardize X and get gamma_hs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.X = X.clone()
        gamma_hs = self._get_gamma_hs(X)
        # print(gamma_hs)
        #Khs = [self._get_kernel_h(X, gamma_h = gammah) for gammah in gamma_hs]

        # delta & alpha
        n = X.shape[0]
        n_train = n * (self.cv - 1) / self.cv
        delta_train = self._get_delta(n_train)
        n_test = n / self.cv
        delta_test = self._get_delta(n_test)
        alpha_scales = self._get_alpha_scales(n)

        # get best (alpha, gamma_h) START
        scores = []
        for it1, (train, test) in enumerate(KFold(n_splits=self.cv).split(np.arange(X.shape[0]).reshape(-1,1))):
            # Standardize X_train
            transX = Scaler()
            X_train = transX.fit_transform(X[train])
            X_test = transX.transform(X[test])
            # Standardize condition_train and get Kf_train, RootKf_train, M_train
            condition_train = Scaler().fit_transform(condition[train])
            Kf_train = self._get_kernel_f(X=condition_train, gamma_f=self._get_gamma_f(condition=condition_train))
            RootKf_train = sqrtm(Kf_train)
            # M_train = RootKf_train @ torch.linalg.inv(
            #     Kf_train / (2 * n_train * (delta_train**2)) + torch.eye(len(train), device=self.device, dtype=X.dtype) / 2) @ RootKf_train
            M_train = RootKf_train @ torch.linalg.lstsq(
                Kf_train / (2 * n_train * (delta_train**2)) + torch.eye(len(train), device=self.device, dtype=X.dtype) / 2, RootKf_train).solution
            # Use M_test based on precomputed RootKf to make sure evaluations are the same
            # M_test = RootKf[np.ix_(test, test)] @ torch.linalg.inv(
            #     Kf[np.ix_(test, test)] / (2 * n_test * (delta_test**2)) + torch.eye(len(test), device=self.device, dtype=X.dtype) / 2) @ RootKf[np.ix_(test, test)]
            M_test = RootKf[np.ix_(test, test)] @ torch.linalg.lstsq(
                Kf[np.ix_(test, test)] / (2 * n_test * (delta_test**2)) + torch.eye(len(test), device=self.device, dtype=X.dtype) / 2, RootKf[np.ix_(test, test)]).solution
            scores.append([])
            for it2, gamma_h in enumerate(gamma_hs):
                Kh_train = self._get_kernel_h(X=X_train, gamma_h=gamma_h)
                KMK_train = Kh_train @ M_train @ Kh_train
                B_train = Kh_train @ M_train @ y[train]
                scores[it1].append([])
                for alpha_scale in alpha_scales:
                    alpha = self._get_alpha(delta_train, alpha_scale)
                    #a = torch.linalg.pinv(KMK_train + alpha * Kh_train) @ B_train
                    a = torch.linalg.lstsq(KMK_train + alpha * Kh_train, B_train).solution
                    res = y[test] - self._get_kernel_h(X=X_test, Y=X_train, gamma_h=gamma_h) @ a
                    scores[it1][it2].append((res.T @ M_test @ res).data[0] / (res.shape[0]**2))

        avg_scores = np.mean(torch.tensor(scores).numpy(), axis=0)
        best_ind = np.unravel_index(np.argmin(avg_scores), avg_scores.shape)
        self.gamma_h = gamma_hs[best_ind[0]]
        self.best_alpha_scale = alpha_scales[best_ind[1]]
        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)
        # M
        # M = RootKf @ torch.linalg.inv(
        #     Kf / (2 * n * delta**2) + torch.eye(n, device=self.device, dtype=X.dtype) / 2) @ RootKf
        M = RootKf @ torch.linalg.lstsq(
            Kf / (2 * n * delta**2) + torch.eye(n, device=self.device, dtype=X.dtype) / 2, RootKf).solution
        # Kh
        Kh = self._get_kernel_h(X, gamma_h=self.gamma_h)

        #self.a = torch.linalg.pinv(Kh @ M @ Kh + self.best_alpha * Kh) @ Kh @ M @ y
        self.a = torch.linalg.lstsq(Kh @ M @ Kh + self.best_alpha * Kh, Kh @ M @ y).solution

        return self

class ApproxRKHSIV(_BaseRKHSIV):

    def __init__(self, n_components=25,
                 gamma_f='auto', gamma_h=0.1,
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto', device='cuda'):
        """
        Parameters:
            n_components : how many approximation components to use
            # kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma_h : the gamma parameter for the kernel of h
            gamma_f : the gamma parameter for the kernel of f
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scale : the scale of the regularization; alpha = alpha_scale * (delta**4)
        """
        self.n_components = n_components
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h 
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale  # regularization strength from Theorem 5
        self.device = device
        # Check cuda GPU device
        if torch.cuda.is_available():
            self.device = self.device
        else:
            self.device = 'cpu'

    def _get_new_approx_instance(self, gamma):
        return Nystroem(gamma=gamma, random_state=1, n_components=self.n_components)

    def fit(self, X, y, condition):
        X         = _to_tensor(X, self.device)
        y         = _to_tensor(y, self.device)
        condition = _to_tensor(condition, self.device)
        eye_n_comp = torch.eye(self.n_components, dtype=X.dtype, device=X.device)
        # Standardize condition and get gamma_f -> RootKf
        condition = Scaler().fit_transform(condition)
        gamma_f = self._get_gamma_f(condition=condition)
        self.gamma_f = gamma_f
        self.featCond = self._get_new_approx_instance(gamma=self.gamma_f)
        RootKf = self.featCond.fit_transform(condition)

        # Standardize X and get gamma_hs -> RootKhs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.featX = self._get_new_approx_instance(gamma=self.gamma_h)
        RootKh = self.featX.fit_transform(X)

        # delta & alpha
        n = X.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        Q = torch.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + eye_n_comp / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + alpha * eye_n_comp)
        B = A @ Q @ RootKf.T @ y
        #self.a = torch.linalg.pinv(W) @ B
        self.a = torch.linalg.lstsq(W, B).solution
        self.fitted_delta = delta
        return self

    def predict(self, X):
        X = _to_tensor(X, self.device)
        X = self.transX.transform(X)
        return self.featX.transform(X) @ self.a

class ApproxRKHSIVCV(ApproxRKHSIV):

    def __init__(self, n_components=25,
                 gamma_f='auto', gamma_hs = 'auto', n_gamma_hs=10,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6, device='cuda'):
        """
        Parameters:
            n_components : how many nystrom components to use
            gamma_f : the gamma parameter for the kernel of f
            gamma_hs : the list of gamma parameters for kernel of h
            n_gamma_hs : how many gamma_hs to try
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**4)
            n_alphas : how mny alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale
        """
        self.n_components = n_components

        self.gamma_f = gamma_f
        self.gamma_hs = gamma_hs
        self.n_gamma_hs=n_gamma_hs

        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  # regularization strength from Theorem 5
        self.n_alphas = n_alphas
        self.cv = cv
        self.device = device
        # Check cuda GPU device
        if torch.cuda.is_available():
            self.device = self.device
        else:
            self.device = 'cpu'

    def _get_gamma_hs(self,X):
        if _check_auto(self.gamma_hs):
            K_X_euclidean = torch.square(torch.cdist(X, X))
            triuInd = torch.triu_indices(K_X_euclidean.size(0),K_X_euclidean.size(0),offset=1)
            #return 1./torch.quantile(K_X_euclidean[triuInd[0],triuInd[1]], torch.linspace(0.1,0.9,steps=self.n_gamma_hs,device=X.device, dtype=X.dtype))/X.shape[1]
            return 1./torch.quantile(K_X_euclidean[triuInd[0],triuInd[1]], torch.linspace(0.1,0.9,steps=self.n_gamma_hs,device=X.device, dtype=X.dtype))
        else:
            return _to_tensor(self.gamma_hs, self.device)

    def fit(self, X, y, condition):
        X         = _to_tensor(X, self.device)
        y         = _to_tensor(y, self.device)
        condition = _to_tensor(condition, self.device)
        eye_n_comp = torch.eye(self.n_components, dtype=X.dtype, device=X.device)
        # Standardize condition and get gamma_f -> RootKf
        self.transcondition = Scaler()
        self.transcondition.fit(condition)
        condition = self.transcondition.transform(condition)
        gamma_f = self._get_gamma_f(condition = condition)
        self.gamma_f = gamma_f
        self.featCond = self._get_new_approx_instance(gamma=gamma_f)
        RootKf = self.featCond.fit_transform(condition)
        self.RootKf = RootKf

        # Standardize X and get gamma_hs -> RootKhs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        gamma_hs = self._get_gamma_hs(X)
        RootKhs = [self._get_new_approx_instance(gamma=gammah).fit_transform(X) for gammah in gamma_hs]

        # delta & alpha
        n = X.shape[0]
        alpha_scales = self._get_alpha_scales(n)
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta_test = self._get_delta(n_test)

        scores = []
        for it1, (train, test) in enumerate(KFold(n_splits=self.cv).split(np.arange(X.shape[0]).reshape(-1,1))):
            RootKf_train, RootKf_test = RootKf[train], RootKf[test]
            Q_train = torch.linalg.pinv(
                RootKf_train.T @ RootKf_train / (2 * n_train * (delta_train**2)) + eye_n_comp / 2)
            Q_test = torch.linalg.pinv(
                RootKf_test.T @ RootKf_test / (2 * n_test * (delta_test**2)) + eye_n_comp / 2)
            scores.append([])
            for it2, RootKh in enumerate(RootKhs):
                RootKh_train, RootKh_test = RootKh[train], RootKh[test]
                A_train = RootKh_train.T @ RootKf_train
                AQA_train = A_train @ Q_train @ A_train.T
                B_train = A_train @ Q_train @ RootKf_train.T @ y[train]
                scores[it1].append([])
                for alpha_scale in alpha_scales:
                    alpha = self._get_alpha(delta_train, alpha_scale)
                    #a = torch.linalg.pinv(AQA_train + alpha *
                    #                   eye_n_comp) @ B_train
                    a = torch.linalg.lstsq(AQA_train + alpha * eye_n_comp, B_train).solution
                    res = RootKf_test.T @ (y[test] - RootKh_test @ a)
                    scores[it1][it2].append((res.T @ Q_test @ res).data[0] / (len(test)**2))

        avg_scores = np.mean(np.array(torch.tensor(scores).numpy()), axis=0)
        best_ind = np.unravel_index(np.argmin(avg_scores), avg_scores.shape)

        self.gamma_h = gamma_hs[best_ind[0]]
        self.featX = self._get_new_approx_instance(gamma=self.gamma_h)
        RootKh = self.featX.fit_transform(X)

        self.best_alpha_scale = alpha_scales[best_ind[1]]
        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)
        print(self.best_alpha_scale, flush=True)

        Q = torch.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + eye_n_comp / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + self.best_alpha * eye_n_comp)
        B = A @ Q @ RootKf.T @ y
        #self.a = torch.linalg.pinv(W) @ B
        self.a = torch.linalg.lstsq(W, B).solution #Faster and numerically stable
        self.fitted_delta = delta
        return self


