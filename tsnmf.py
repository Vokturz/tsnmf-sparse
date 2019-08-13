import numpy as np
import scipy.sparse as sp
from math import sqrt
import numbers
import time
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


def create_constraint_matrix(labels, n_components):
    L = np.ones((len(labels), n_components)) * 1  # initialize matrix of ones

    for document_index, topic_index_list in enumerate(labels):
        if len(topic_index_list) > 0:  # if document has been labeled
            L[document_index, :] = 0  # set all labels to zero for that doc initially

            for topic_index in topic_index_list:
                L[document_index, topic_index] = 1  # set labeled topic / document to 1
    return L

def norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return sqrt(squared_norm(x))

def _check_init(A, shape, whom):
    A = check_array(A)
    if np.shape(A) != shape:
        raise ValueError('Array with wrong shape passed to %s. Expected %s, '
                         'but got %s ' % (whom, shape, np.shape(A)))
    check_non_negative(A, whom)
    if np.max(A) == 0:
        raise ValueError('Array passed to %s is full of zeros.' % whom)

def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T).
    Parameters
    ----------
    X : array-like
        First matrix
    Y : array-like
        Second matrix
    """
    return np.dot(X.ravel(), Y.ravel())

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

    def fit(self, X, labels, y=None, **params):
        """Learn a NMF model for the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        Returns
        -------
        self
        """
        self.fit_transform(X, labels,**params)
        return self

    def fit_transform(self, X, labels, y=None, W=None, H=None):
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
        W, H, n_iter = topic_supervised_factorization(X, W, H,self.n_components, labels, init =self.init,
                                        tol=self.tol, max_iter=self.max_iter, verbose=self.verbose)
                                        
        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        return W

    def transform(self, X, labels):
        W, H, n_iter = topic_supervised_factorization(X, W=None, H=self.components_,
                                    n_components = self.n_components_, labels= labels, init =self.init,
                                        update_H=False,tol=self.tol, max_iter=self.max_iter, verbose=self.verbose)
        return W        

    
def topic_supervised_factorization(X, W=None, H=None, n_components=None,
                                    labels=None, init=None, update_H=True,
                                    tol=1e-4, max_iter=200, regularization=None,
                                    random_state=None, verbose=0):
    """
        update_H : boolean, default: True
            Set to True, both W and H will be estimated from initial guesses.
            Set to False, only W will be estimated.

        regularization : 'both' | 'components' | 'transformation' | None
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

    # labels will be list
    if not isinstance(labels, list):
        raise ValueError()

    if not update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        # 'mu' solver should not be initialized by zeros
        avg = np.sqrt(X.mean() / n_components)
        W = np.full((n_samples, n_components), avg)
        W = W
    else:
        W, H = _initialize_tsnmf(X, n_components, init=init, random_state=random_state)
    L = create_constraint_matrix(labels, n_components)
    with np.errstate(invalid='ignore'):
        W, H, n_iter = _fit_multiplicative_update(X, W, H, L, max_iter, tol,
                                                    update_H, verbose)
    return W, H, n_iter

def _fit_multiplicative_update(X, W, H, L, max_iter=200, tol=1e-4,
                                update_H=True, verbose=0):
    error_at_init = _beta_divergence(X, W, H, L, square_root=True)
    previous_error = error_at_init
    HHt, XHt, = None, None
    for n_iter in range(1,max_iter + 1):
        # update W
        delta_W, HHt, XHt= _multiplicative_update_w(
                            X, W, H, L, HHt, XHt, update_H)
        W *= delta_W

        # update H
        if update_H:
            delta_H = _multiplicative_update_h(X, W, H, L)
            H *= delta_H

            HHt, XHt = None, None

        if tol > 0: #and n_iter % 10 == 0:
            error = _beta_divergence(X, W, H, L, square_root=True)

            #if verbose:
            #    iter_time = time.time()
            #    print("Epoch %02d reached after %.3f seconds, error: %f" %
            #          (n_iter, iter_time - start_time, error))

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    return W, H, n_iter

def _multiplicative_update_w(X, W, H, L,  HHt=None,
                            XHt=None, update_H=True):

    # assuming Frobenius norm
    # Numerator
    if XHt is None:
        XHt = safe_sparse_dot(X,H.T)
    if update_H:
        numerator = XHt
    else:
        numerator = XHt.copy()
    numerator *= L

    # Denominator
    if HHt is None:
        HHt = np.dot(H,H.T)
    WoL = W*L
    denominator = np.dot(WoL,HHt)
    denominator *= L
    numerator /= denominator
    # numerator.data /= np.array(denominator[numerator.nonzero()])[0]
    delta_W = np.nan_to_num(numerator)
    return delta_W, HHt, XHt

def _multiplicative_update_h(X, W, H, L):

    # Assuming Frobenius norm

    # Numerator
    WoL = W*L
    numerator = safe_sparse_dot(WoL.T,X)

    # Denominator
    denominator = np.dot(np.dot(WoL.T,WoL),H)

    numerator /= denominator
    # numerator.data /= np.array(denominator[numerator.nonzero()])[0]
    delta_H = np.nan_to_num(numerator)
    return delta_H

def _beta_divergence(X, W, H, L, square_root=False):
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Avoid the creation of the dense np.dot(W, H) if X is sparse.
    if sp.issparse(X):
        norm_X = np.dot(X.data, X.data)
        WoL = W*L
        norm_WoLH = trace_dot(np.dot(np.dot(WoL.T,WoL),H),H)
        cross_prod = trace_dot((X * H.T), WoL)
        res = (norm_X + norm_WoLH - 2. * cross_prod) / 2
    else:
        WoL = W*L
        res = squared_norm(X- np.dot(WoL,H)) / 2
    
    if square_root:
        return np.sqrt(res * 2)
    else:
        return res
