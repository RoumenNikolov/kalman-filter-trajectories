import numpy as np
from state_logger import StateLogger

class KalmanFilter:
    def __init__(self, F, Q, H, R, logger=None,
                 F_builder=None, Q_builder=None,
                 B=None):
        """
        2D Kalman filter with optional variable-dt and control input support.

        Parameters
        ----------
        F : (n, n) ndarray
            State transition matrix (used when dt is not supplied to predict()).
        Q : (n, n) ndarray
            Process noise covariance (used when dt is not supplied to predict()).
        H : (m, n) ndarray
            Measurement matrix.  z = H s + noise.
        R : (m, m) ndarray
            Measurement noise covariance.
        logger : StateLogger or None
            Optional recorder for s, P, K, z, z_hat, e.
        F_builder : callable(dt) -> (n, n) ndarray, optional
            If supplied, predict() calls F_builder(dt) instead of self.F.
            Required for variable-dt tracking (e.g. GPS).
        Q_builder : callable(dt) -> (n, n) ndarray, optional
            Same as F_builder but for Q. Supply together with F_builder.
        B : (n, p) ndarray or None
            Control input matrix.
            s_{k+1|k} = F s_{k|k} + B u_k
        """
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.B = B
        self.logger     = logger
        self._F_builder = F_builder
        self._Q_builder = Q_builder

    def predict(self, s_k_k, P_k_k, dt=None, u=None):
        """
        Kalman predict step.

            s_{k+1|k} = F s_{k|k} + B u_k
            P_{k+1|k} = F P_{k|k} F^T + Q

        Parameters
        ----------
        s_k_k  : (n, 1) ndarray   current posterior state
        P_k_k  : (n, n) ndarray   current posterior covariance
        dt     : float or None
            If provided and F_builder was given at construction,
            F and Q are rebuilt for this time step.
        u      : (p, 1) ndarray or None
            Control input. Applied only if self.B is not None.

        Returns
        -------
        s_k1_k : (n, 1) ndarray
        P_k1_k : (n, n) ndarray
        """
        # --- build F and Q ---
        if dt is not None and self._F_builder is not None:
            F = self._F_builder(dt)
            Q = self._Q_builder(dt) if self._Q_builder is not None else self.Q
        else:
            F = self.F
            Q = self.Q

        # --- state prediction ---
        s_k1_k = F @ s_k_k
        if self.B is not None and u is not None:
            s_k1_k = s_k1_k + self.B @ u

        # --- covariance prediction ---
        P_k1_k = F @ P_k_k @ F.T + Q

        return s_k1_k, P_k1_k

    def update(self, s_k1_k, P_k1_k, z_k1):
        """
        Kalman update step.

            z_hat_{k+1|k} = H s_{k+1|k}
            e_{k+1}       = z_{k+1} - z_hat_{k+1|k}
            S_{k+1}       = H P_{k+1|k} H^T + R
            K_{k+1}       = P_{k+1|k} H^T S_{k+1}^{-1}
            s_{k+1|k+1}   = s_{k+1|k} + K_{k+1} e_{k+1}
            P_{k+1|k+1}   = (I - K_{k+1} H) P_{k+1|k}

        Parameters
        ----------
        s_k1_k  : (n, 1) ndarray   predicted state
        P_k1_k  : (n, n) ndarray   predicted covariance
        z_k1    : (m, 1) ndarray   measurement

        Returns
        -------
        s_k1_k1 : (n, 1) ndarray   updated state
        P_k1_k1 : (n, n) ndarray   updated covariance
        """
        # --- innovation ---
        z_hat = self.H @ s_k1_k
        e     = z_k1 - z_hat

        # --- Kalman gain ---
        S = self.H @ P_k1_k @ self.H.T + self.R
        K = P_k1_k @ self.H.T @ np.linalg.inv(S)

        # --- state update ---
        s_k1_k1 = s_k1_k + K @ e

        # --- covariance update ---
        n       = P_k1_k.shape[0]
        P_k1_k1 = (np.eye(n) - K @ self.H) @ P_k1_k

        # --- log ---
        if self.logger is not None:
            self.logger.append(
                s=s_k1_k1, P=P_k1_k1,
                K=K, z=z_k1, z_hat=z_hat, e=e
            )

        return s_k1_k1, P_k1_k1

    def reset(self):
        """
        Clear the logger history. Useful when running the filter
        over multiple segments with the same instance.
        """
        if self.logger is not None:
            for key in self.logger.data:
                self.logger.data[key] = []