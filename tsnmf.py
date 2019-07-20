import numpy as np
from scipy.sparse as sp

class TSNMF:
    r"""
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


