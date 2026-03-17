import numpy as np
import matplotlib.pyplot as plt



class GPSStats:
    """
    Statistics and visualisation for a single Kalman-filtered GPS segment.

    Computes acceleration RMS (from positions and from state vector),
    innovation statistics, and provides trajectory and innovation plots.

    Parameters
    ----------
    s_hist  : ndarray  State history, shape (N, 1, n) or (N, n).
    e_hist  : ndarray  Innovation history, shape (N, 1, 2) or (N, 2).
    z_hist  : ndarray  Measurement history, shape (N, 1, 2) or (N, 2).
    x       : ndarray  Raw GPS x positions in metres, shape (N,).
    y       : ndarray  Raw GPS y positions in metres, shape (N,).
    dt_arr  : ndarray  Time steps in seconds, length N-1.
    model   : str      Motion model identifier — 'cv' or 'ca'.
    """

    def __init__(self, s_hist, e_hist, z_hist, x, y, dt_arr, model='cv'):
        self.s_hist  = np.squeeze(s_hist)  # (N, n)
        self.e_hist  = np.squeeze(e_hist)  # (N, 2)
        self.z_hist  = np.squeeze(z_hist)  # (N, 2)
        self.x       = x
        self.y       = y
        self.dt_arr  = dt_arr
        self.model   = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kf_xy(self):
        """Return Kalman-filtered x, y positions from state history."""
        return self.s_hist[:, 0], self.s_hist[:, 1]

    # ------------------------------------------------------------------
    # Acceleration RMS
    # ------------------------------------------------------------------

    @staticmethod
    def _acceleration_rms_pos(x, y, dt_arr):
        """
        Compute acceleration RMS from position arrays via finite differences.

        Velocity     : vx = diff(x) / dt
        Acceleration : ax = diff(vx) / dt[1:]
        RMS          : sqrt(mean(ax^2 + ay^2))

        Used as a noise indicator — lower values mean a smoother trajectory.

        Parameters
        ----------
        x, y    : ndarray  Position arrays in metres.
        dt_arr  : ndarray  Time steps in seconds, length len(x) - 1.

        Returns
        -------
        float  Acceleration RMS in m/s^2.
        """
        vx = np.diff(x) / dt_arr
        vy = np.diff(y) / dt_arr
        ax = np.diff(vx) / dt_arr[1:]
        ay = np.diff(vy) / dt_arr[1:]
        return np.sqrt(np.mean(ax**2 + ay**2))

    def acc_rms_raw(self):
        """
        Acceleration RMS of the raw GPS positions.

        Returns
        -------
        float  Acceleration RMS in m/s^2.
        """
        return self._acceleration_rms_pos(self.x, self.y, self.dt_arr)

    def acc_rms_kf_pos(self):
        """
        Acceleration RMS of the Kalman-filtered positions.

        Extracts x, y from the state history and applies finite differences —
        comparable to acc_rms_raw() across both CV and CA.

        Returns
        -------
        float  Acceleration RMS in m/s^2.
        """
        kx, ky = self._kf_xy()
        return self._acceleration_rms_pos(kx, ky, self.dt_arr[1:])

    def acceleration_rms_state(self):
        """
        Acceleration RMS directly from the CA state vector (ax, ay components).

        Only valid for the CA model (n=6), where acceleration is explicitly
        estimated by the filter. Returns None for CV.

        Returns
        -------
        float  Acceleration RMS in m/s^2, or None if model != 'ca'.
        """
        if self.model != 'ca':
            return None
        ax = self.s_hist[:, 4]
        ay = self.s_hist[:, 5]
        return np.sqrt(np.mean(ax**2 + ay**2))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self):
        """
        Compute all evaluation metrics for this segment.

        Returns
        -------
        dict with keys:
            model        : str     Motion model ('cv' or 'ca').
            acc_raw      : float   Acceleration RMS of raw GPS (m/s^2).
            acc_kf_pos   : float   Acceleration RMS of KF positions (m/s^2).
            acc_kf_state : float   Acceleration RMS from CA state vector (m/s^2),
                                   or None for CV.
            rse_median   : float   Median innovation norm in metres.
            rse_p95      : float   95th percentile innovation norm in metres.
            e            : ndarray Innovation array, shape (N, 2).
        """
        e_norm = np.sqrt(self.e_hist[:, 0]**2 + self.e_hist[:, 1]**2)

        return {
            'model':        self.model,
            'acc_raw':      self.acc_rms_raw(),
            'acc_kf_pos':   self.acc_rms_kf_pos(),
            'acc_kf_state': self.acceleration_rms_state(),
            'rse_median':   np.median(e_norm),
            'rse_p95':      np.percentile(e_norm, 95),
            'e':            self.e_hist,
        }

   