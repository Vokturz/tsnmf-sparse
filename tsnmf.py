import numpy as np
import scipy.sparse as sp
from math import sqrt
import numbers
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_non_negative

INTEGER_TYPES = (numbers.Integral, np.integer)

def _initialize_tsnmf(X, n_components, init=None, eps=1e-6, random_state=None):
    """
    Initialization of matrices W and H (X = WH)

    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH
    H : array-like, shape (n_components, n_features)
        Initial guesses for solving X ~= WH
    """

    check_non_negative(X, "TSNMF initialization")
    n_samples, n_features = X.shape
    if (init is not None and init != 'random'
            and n_components > min(n_samples, n_features)):
            raise ValueError("init = '{}' can only be used when "
                         "n_components <= min(n_samples, n_features)"
                         .format(init))
    
    if init is None:
        if n_components <= min(n_samples, n_features):
            init = 'nndsvd'
        else:
            init = 'random'
   
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features)
        W = avg * rng.randn(n_samples, n_components)
        # we do not write np.abs(H, out=H) to stay compatible with
        # numpy 1.5 and earlier where the 'out' keyword is not
        # supported as a kwarg on ufuncs
        np.abs(H, H)
        np.abs(W, W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd',)))

    return W, H

def norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return sqrt(squared_norm(x))

class TSNMF:
    """
    Parameters
    ----------
    n_components: int or None
        Number of components (topics)
    
    init = None | 'random' | 'nndsvd'
        Method used to initialize the procedure
        Defaul: None.
        Valid options:

        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise random.
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : bool, default=False
        Whether to be verbose.
    """
    def __init__(self, n_components=None, init=None, tol=1e-4,
                 max_iter=200, random_state=None, verbose=0):
        
        self.n_components = n_components
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose


    def fit_transform(self, X, y=None, W=None, H=None):
        return
    
def topic_supervised_factorization(X, W=None, H=None, n_components=None,
                                    init=None, update_H=True, tol=1e-4,
                                    max_iter=200, regularization=None,
                                    random_state=None, verbose=0):
    """
        update_H : boolean, default: True
            Set to True, both W and H will be estimated from initial guesses.
            Set to False, only W will be estimated.

        rgularization : 'both' | 'components' | 'transformation' | None
            Select whether the regularization affects the components (H), the
            transformation (W), both or none of them.
    """

    X = check_array(X, accept_sparse=('csr','csc'), dtype=float) # from utils
    check_non_negative(X, "TSNMF (input X)") # from utils.validation
    
    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    #Validation from NMF sklearn source code
    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    W, H = _initialize_tsnmf(X, n_components, init=init, random_state=random_state)

    W, H, n_iter = _fit_multiplicative_update(X, W, H, max_iter, tol,
                                                update_H, verbose)


def _fit_multiplicative_update(X, W, H, max_iter=200, tol=1e-4, update_H=True,
                                verbose=0):
    return W, H, n_iter
