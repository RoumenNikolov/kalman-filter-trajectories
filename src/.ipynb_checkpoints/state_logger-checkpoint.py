import numpy as np

class StateLogger:
    """
    Flexible logger for Kalman filter quantities.

    This class stores sequences of state estimates, covariance matrices,
    and (optionally) Kalman gains in Python lists, and can convert them
    to NumPy arrays when needed.

    Parameters
    ----------
    log_x : bool, optional
        If True, log state vectors x_k (default: True).
    log_P: bool, optional
        If True, log covariance matrices P_k (default: False).
    log_K: bool, optional
        If True, log Kalman gains K_k (default: False).
    """

    def __init__(self, log_x: bool = True, log_P: bool = False, log_K: bool = False):
        self.log_x = log_x
        self.log_P = log_P
        self.log_K = log_K

        # Lists for dynamic logging; shapes are inferred from the first entry
        if log_x:
            self.X = []      # list of state vectors (n, 1) or (n,)
        if log_P:
            self.P = []      # list of covariance matrices (n, n)
        if log_K:
            self.K = []      # list of Kalman gains (n, m)

    def append(self, x=None, P=None, K=None):
        """
        Append a new set of logged quantities for the current time step.

        Parameters
        ----------
        x : np.ndarray, optional
            State estimate at current step (typically shape (n, 1) or (n,)).
        P : np.ndarray, optional
            State covariance matrix at current step (shape (n, n)).
        K : np.ndarray, optional
            Kalman gain at current step (shape (n, m)).
        """
        if self.log_x and x is not None:
            self.X.append(np.array(x, copy=True))
        if self.log_P and P is not None:
            self.P.append(np.array(P, copy=True))
        if self.log_K and K is not None:
            self.K.append(np.array(K, copy=True))

    def as_arrays(self):
        """
        Convert logged lists to NumPy arrays.

        Returns
        -------
        X: np.ndarray or None
           An array of logged states, shape (n, T), where T is the number
           of logged steps, or None if log_x is False.
        P: np.ndarray or None
           Array of logged covariances, shape (n, n, T), or None if log_P is False.
        K: np.ndarray or None
           Array of logged Kalman gains, shape (n, m, T), or None if log_K is False.
        """
        X = np.column_stack(self.X) if self.log_x and self.X else None
        P = np.stack(self.P, axis=2) if self.log_P and self.P else None
        K = np.stack(self.K, axis=2) if self.log_K and self.K else None
        return X, P, K
