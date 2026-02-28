import numpy as np

class Kalman2D:
    def __init__(self, F, Q, H, R, logger = None):
        """
        Parameters
        ----------
        F : (n, n) ndarray
            State transition matrix.
        Q : (n, n) ndarray
            Process noise covariance.
        H : (m, n) ndarray
            Measurement matrix, maps nD state to mD position:
                z = H s + m.
        R : (m, m) ndarray
            Measurement noise covariance.
        logger - Optionally uses a StateLogger instance to record states, covariances, and gains.
        """
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

        self.logger = logger     # StateLogger or None

    def predict(self, s_k_k, P_k_k):
        """
        Kalman predict step for nD state.
    
        Implements:
            s_{k+1|k} = F s_{k|k}
            P_{k+1|k} = F P_{k|k} F^T + Q
    
        Parameters
        ----------
        s_k_k : (n, 1) ndarray
            Current posterior state s_{k|k}.
        P_k_k : (n, n) ndarray
            Current posterior covariance P_{k|k}.
        F : (n, n) ndarray
            State transition matrix.
        Q : (n, n) ndarray
            Process noise covariance.
    
        Returns
        -------
        s_k1_k : (n, 1) ndarray
            Predicted state s_{k+1|k}.
        P_k1_k : (n, n) ndarray
            Predicted covariance P_{k+1|k}.
        """
        s_k1_k = self.F @ s_k_k
        P_k1_k = self.F @ P_k_k @ self.F.T + self.Q
        return s_k1_k, P_k1_k

    def z_innov(self, s_k1_k, z_k1):
        """
        Measurement model and innovation for mD position sensor.
    
        Parameters
        ----------
        s_k1_k : (n, 1) ndarray
            Predicted state s_{k+1|k}.
        z_k1 : (m, 1) ndarray
            Actual measurement z_{k+1} from the sensor.

        Internal
        --------
        H: Measurement matrix, maps nD state to mD position:
           z = H s + m.
           
        Returns
        -------
        ẑ_k1_k : (m, 1) ndarray
            Predicted measurement ẑ_{k+1|k} = H s_{k+1|k}.
        e_k1 : (m, 1) ndarray
            Innovation e_{k+1} = z_{k+1} - ž_{k+1|k}.
        """
        # Predicted measurement: ẑ_{k+1|k} = H s_{k+1|k}
        z_k1_k = self.H @ s_k1_k
    
        # Innovation (surprise): e_{k+1} = z_{k+1} - ž_{k+1|k}
        e_k1 = z_k1 - self.H @ s_k1_k
        e_k1 = z_k1 - z_k1_k
    
        return z_k1_k, e_k1

    def kalman_gain(self, P_k1_k):
        """
        Kalman gain:
            K_{k+1} = P_{k+1|k} H^T (H P_{k+1|k} H^T + R)^{-1}
    
        Parameters
        ----------
        P_k1_k : (n, n) ndarray
            Predicted state covariance P_{k+1|k}.

        Internal
        --------
        H: Measurement matrix, maps nD state to mD position:
           z = H s + m.
    
        Returns
        -------
        K_k1 : (n, m) ndarray
            Kalman gain K_{k+1}.
        S_k1 : (m, m) ndarray
            Innovation covariance S_{k+1} = H P_{k+1|k} H^T + R.
        """
        # Innovation covariance in measurement space:
        # S_{k+1} = H P_{k+1|k} H^T + R
        S_k1 = self.H @ P_k1_k @ self.H.T + self.R
    
        # Kalman gain:
        # K_{k+1} = P_{k+1|k} H^T S_{k+1}^{-1}
        K_k1 = P_k1_k @ self.H.T @ np.linalg.inv(S_k1)
    
        return K_k1, S_k1


    def state_update(self, s_k1_k, e_k1, K_k1):
        """
        Kalman state update with innovation:
    
            e_{k+1}      = z_{k+1} - H s_{k+1|k}
            s_{k+1|k+1}  = s_{k+1|k} + K_{k+1} e_{k+1}
    
        Parameters
        ----------
        s_k1_k : (n, 1) ndarray
            Predicted state s_{k+1|k}.

        K_k1 : (n, m) ndarray
            Kalman gain K_{k+1}.
    
        Returns
        -------
        s_k1_k1 : (n, 1) ndarray
            Updated state estimate s_{k+1|k+1}.
        """
        # Updated state: s_{k+1|k+1} = s_{k+1|k} + K_{k+1} e_{k+1}
        s_k1_k1 = s_k1_k + K_k1 @ e_k1
    
        return s_k1_k1

    def cov_update(self, P_k1_k, K_k1):
        """
        Kalman covariance update:
    
            P_{k+1|k+1} = (I - K_{k+1} H) P_{k+1|k}
    
        Parameters
        ----------
        P_k1_k : (n, n) ndarray
            Predicted covariance P_{k+1|k}.
        K_k1 : (n, m) ndarray
            Kalman gain K_{k+1}.
    
        Returns
        -------
        P_k1_k1 : (n, n) ndarray
            Updated covariance P_{k+1|k+1}.
        """
        n = P_k1_k.shape[0]
        I = np.eye(n)
        P_k1_k1 = (I - K_k1 @ self.H) @ P_k1_k
        return P_k1_k1