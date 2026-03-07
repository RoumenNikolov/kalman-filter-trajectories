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
        float  Acceleration RMS in m/s².
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
        float  Acceleration RMS in m/s².
        """
        return self._acceleration_rms_pos(self.x, self.y, self.dt_arr)

    def acc_rms_kf_pos(self):
        """
        Acceleration RMS of the Kalman-filtered positions.

        Extracts x, y from the state history and applies finite differences —
        comparable to acc_rms_raw() across both CV and CA.

        Returns
        -------
        float  Acceleration RMS in m/s².
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
        float  Acceleration RMS in m/s², or None if model != 'ca'.
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
            acc_raw      : float   Acceleration RMS of raw GPS (m/s²).
            acc_kf_pos   : float   Acceleration RMS of KF positions (m/s²).
            acc_kf_state : float   Acceleration RMS from CA state vector (m/s²),
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

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_innovations(self, title=''):
        """
        Plot innovation time series and distributions for ex and ey.

        A well-tuned filter produces zero-mean, approximately Gaussian
        innovations with std ≈ sigma_r. Prints summary statistics.

        Layout: 2x2 grid — time series (top), histograms (bottom).

        Parameters
        ----------
        title : str  Optional plot title.
        """
        ex, ey = self.e_hist[:, 0], self.e_hist[:, 1]

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))

        axes[0, 0].plot(ex, 'b-', lw=0.8, label='innovation ex')
        axes[0, 0].axhline(0, color='r', ls='--', label='zero mean')
        axes[0, 0].set_title('Innovation ex'); axes[0, 0].legend(); axes[0, 0].grid(True)

        axes[0, 1].plot(ey, 'b-', lw=0.8, label='innovation ey')
        axes[0, 1].axhline(0, color='r', ls='--', label='zero mean')
        axes[0, 1].set_title('Innovation ey'); axes[0, 1].legend(); axes[0, 1].grid(True)

        axes[1, 0].hist(ex, bins=30, color='steelblue', alpha=0.8)
        axes[1, 0].axvline(ex.mean(), color='r', ls='--', label=f'mean={ex.mean():.2f}')
        axes[1, 0].set_title(f'ex distribution  σ={ex.std():.2f} m')
        axes[1, 0].legend(); axes[1, 0].grid(True)

        axes[1, 1].hist(ey, bins=30, color='steelblue', alpha=0.8)
        axes[1, 1].axvline(ey.mean(), color='r', ls='--', label=f'mean={ey.mean():.2f}')
        axes[1, 1].set_title(f'ey distribution  σ={ey.std():.2f} m')
        axes[1, 1].legend(); axes[1, 1].grid(True)

        plt.suptitle(title or f'Innovation analysis — {self.model.upper()}')
        plt.tight_layout()
        plt.show()

        print(f"ex: mean={ex.mean():.3f}  std={ex.std():.3f}  min={ex.min():.3f}  max={ex.max():.3f}")
        print(f"ey: mean={ey.mean():.3f}  std={ey.std():.3f}  min={ey.min():.3f}  max={ey.max():.3f}")

    def plot_trajectory(self, title='', zoom=None, ax=None):
        """
        Plot raw GPS vs Kalman filter estimate in local Cartesian coordinates.

        Parameters
        ----------
        title : str        Optional plot title.
        zoom  : tuple|None Optional axis limits (x_min, x_max, y_min, y_max) in km.
        """
        kx, ky = self._kf_xy()
        mx, my = self.z_hist[:, 0], self.z_hist[:, 1]
    
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
    
        ax.plot(mx/1000, my/1000, 'o-', color='crimson',
                ms=2, lw=1, alpha=0.7, label='raw GPS')
        ax.plot(kx/1000, ky/1000, 'b-', lw=1.5, label='KF estimate')
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.grid(True); ax.legend()
        ax.set_title(title or f"{self.model.upper()} (n={len(kx)})")
    
        if zoom:
            ax.set_xlim(zoom[0], zoom[1])
            ax.set_ylim(zoom[2], zoom[3])
    
        if show:
            plt.tight_layout()
            plt.show()

    def plot_raw_trajectory(self, title=''):
        """
        Plot raw GPS trajectory and step size distribution.

        Left panel  : 2D path in local Cartesian coordinates (km).
        Right panel : histogram of step sizes between consecutive fixes,
                      with mean and std — used to estimate sigma_r.

        Parameters
        ----------
        title : str  Optional plot title.
        """
        steps = np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(self.x/1000, self.y/1000, 'o-', color='crimson',
                     ms=2, lw=1, alpha=0.7)
        axes[0].set_xlabel('X (km)'); axes[0].set_ylabel('Y (km)')
        axes[0].set_title(f"Raw GPS (n={len(self.x)})")
        axes[0].grid(True)

        axes[1].hist(steps, bins=30, color='crimson', alpha=0.8)
        axes[1].axvline(steps.mean(), color='white', lw=2, ls='--',
                        label=f'mean={steps.mean():.1f} m')
        axes[1].set_xlabel('Step size (m)'); axes[1].set_ylabel('Count')
        axes[1].set_title(f"Step distribution  σ={steps.std():.2f} m")
        axes[1].legend(); axes[1].grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()