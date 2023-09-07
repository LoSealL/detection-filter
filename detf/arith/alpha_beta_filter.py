"""
Copyright Wenyi Tang 2023

:Author: Wenyi Tang
:Email: wenyitang@outlook.com

Implementation of vectorized alpha-beta filter
https://en.wikipedia.org/wiki/Alpha_beta_filter
"""

from typing import Any

import numpy as np


class AlphaBetaFilter:
    """1-st order velocity filter"""

    def __init__(self, alpha: float, beta: float, timescale: float = 1.0):
        if alpha > 1.0 or alpha < 0:
            raise ValueError(f"alpha must belong to [0, 1], got {alpha}.")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}.")
        if timescale <= 0:
            raise ValueError(f"timescale must be positive, got {timescale}.")
        self._alpha = alpha
        self._beta = beta
        self._t = 0
        self.timescale = timescale
        self.state = None
        self.velo = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.filter(*args, **kwds)

    def filter(
        self,
        value: np.ndarray,
        *,
        alpha: float = None,
        beta: float = None,
        t: float = None,
    ):
        """_summary_

        Args:
            value (np.ndarray): _description_
            alpha (float, optional): _description_. Defaults to None.
            beta (float, optional): _description_. Defaults to None.
            t (float, optional): _description_. Defaults to None.
        """
        alpha = self._alpha if alpha is None else np.array(alpha)
        beta = self._beta if beta is None else np.array(beta)
        dt = 1.0 if t is None else (t - self._t) * self.timescale
        if self.state is None:
            self.state = value
            self.velo = np.zeros_like(value, dtype=np.float32)
        else:
            pred = self.state + self.velo * dt
            self.state = alpha * value + (1 - alpha) * pred
            self.velo += beta * (value - pred) / dt
        return self.state
