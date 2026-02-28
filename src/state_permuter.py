import numpy as np

class StatePermuter:
    """
    Helper for reordering state vectors and system matrices.

    Stores permutation matrix P such that:
        x_new = P @ x_old
    """

    def __init__(self, old_order, new_order):
        """
        Parameters
        ----------
        old_order : list[str]
            State names in original order, e.g. ["x","y","vx","vy","ax","ay"].
        new_order : list[str]
            State names in desired order, e.g. ["x","vx","ax","y","vy","ay"]
        where:
            "x" – position along the x‑axis (horizontal position).
            "vx" – velocity along the x‑axis, i.e. x‑component of velocity.
            "ax" – acceleration along the x‑axis, i.e. x‑component of acceleration
            "y" – position along the y‑axis (vertical or second horizontal axis, depending on your frame).
            "vy" – velocity along the y‑axis, i.e. y‑component of velocity.
            "ay" – acceleration along the y‑axis, i.e. y‑component of acceleration.
        """
        self.old_order = list(old_order)
        self.new_order = list(new_order)
        n = len(old_order)
        P = np.zeros((n, n))
        for new_idx, name in enumerate(new_order):
            old_idx = old_order.index(name)
            P[new_idx, old_idx] = 1.0
        self.P = P
        self.P_T = P.T  # useful for H, and for going back

    def state_to_new(self, x_old):
        """x_new = P @ x_old."""
        return self.P @ x_old

    def state_to_old(self, x_new):
        """x_old = P^T @ x_new (inverse permutation)."""
        return self.P_T @ x_new

    def system_to_new(self, F_old, Q_old, H_old):
        """
        Convert (F,Q,H) from old to new state coordinates.
        """
        F_new = self.P @ F_old @ self.P_T
        Q_new = self.P @ Q_old @ self.P_T
        H_new = H_old @ self.P_T
        return F_new, Q_new, H_new

    def P_back(self):
        """Return inverse permutation matrix (same as P^T)."""
        return self.P_T
