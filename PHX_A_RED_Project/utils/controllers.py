"""
ShiftingKappaController — single source of truth.

Live PI-controller that adjusts ARED's kappa ("paranoia") so that the
long-term running average of queries per 100 points trends toward a target.

This is the authoritative copy (merged from the best parts of the two
shifting-kappa spectrogram runners).
"""
from collections import deque
from typing import Optional


class ShiftingKappaController:
    """
    Adjusts ared.kappa after every point (or every N points) using a
    proportional + integral controller on the observed query rate.

    Kappa semantics:
        higher kappa → stricter anomaly test → fewer queries
        lower kappa  → more queries (higher paranoia)

    The controller *decreases* kappa when rate > target,
    *increases* when rate < target.
    """

    def __init__(
        self,
        target_queries_per_100: float = 4.0,
        window_size: int = 500,
        min_kappa: float = 0.1,
        max_kappa: float = 8.0,
        adjust_every: int = 20,
        warmup_points: int = 50,
        initial_kappa: Optional[float] = None,
        verbose: bool = False,
        use_ema: bool = True,
        ema_span: int = 500,
        # PI controller
        kp: float = 0.65,
        ki: float = 0.04,
        kd: float = 0.0,
        max_step: float = 0.07,
        min_step: float = 0.003,
        deadband: float = 0.012,
        keep_history: bool = True,
    ):
        self.target = float(target_queries_per_100)
        self.window = deque(maxlen=window_size)
        self.min_k = min_kappa
        self.max_k = max_kappa
        self.adjust_every = adjust_every
        self.warmup_points = warmup_points
        self.verbose = verbose
        self.keep_history = keep_history

        # EMA
        self.use_ema = use_ema
        self.ema_alpha = 2.0 / (ema_span + 1.0) if (use_ema and ema_span > 0) else 0.0
        self.ema_fraction = 0.0

        # PI state
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_step = max_step
        self.min_step = min_step
        self.deadband = deadband
        self.integral = 0.0
        self.prev_error = 0.0

        self.points = 0
        self.current_rate = 0.0
        self.total_queries = 0
        self.kappa = initial_kappa if initial_kappa is not None else 1.0
        self.history = [] if keep_history else None
        self.min_kappa_seen = self.kappa
        self.max_kappa_seen = self.kappa

    def record_and_adjust(self, ared, did_query: bool):
        """Call after every process_point (or first_point)."""
        self.window.append(1 if did_query else 0)
        self.points += 1
        if did_query:
            self.total_queries += 1

        wlen = max(1, len(self.window))
        window_fraction = sum(self.window) / wlen

        if self.use_ema:
            is_q = 1.0 if did_query else 0.0
            self.ema_fraction = (1.0 - self.ema_alpha) * self.ema_fraction + self.ema_alpha * is_q
            control_fraction = self.ema_fraction
        else:
            control_fraction = window_fraction

        rate_per_100 = control_fraction * 100.0
        self.current_rate = rate_per_100

        if self.keep_history and self.history is not None:
            self.history.append((self.points, rate_per_100, ared.kappa))

        k = ared.kappa
        if k < self.min_kappa_seen:
            self.min_kappa_seen = k
        if k > self.max_kappa_seen:
            self.max_kappa_seen = k

        if (self.points >= self.warmup_points and
                self.points % self.adjust_every == 0):

            target_fraction = self.target / 100.0
            error = control_fraction - target_fraction

            old_kappa = ared.kappa

            if abs(error) > self.deadband:
                self.integral += error
                self.integral = max(-8.0, min(8.0, self.integral))

                derivative = error - self.prev_error

                raw_delta = -(self.kp * error + self.ki * self.integral + self.kd * derivative)

                step_size = abs(raw_delta)
                step_size = max(self.min_step, min(self.max_step, step_size))

                if error > 0:
                    # too many queries → reduce paranoia
                    new_kappa = max(self.min_k, old_kappa - step_size)
                else:
                    new_kappa = min(self.max_k, old_kappa + step_size)

                ared.kappa = new_kappa

            self.prev_error = error

            if self.verbose and abs(ared.kappa - old_kappa) > 1e-9:
                direction = "↑" if ared.kappa > old_kappa else "↓"
                print(f"    [ShiftingKappa] paranoia {direction} {old_kappa:.3f} → {ared.kappa:.3f} "
                      f"(rate={rate_per_100:.1f}/100, target={self.target})")

        self.kappa = ared.kappa

    def get_current_rate(self) -> float:
        return self.current_rate

    def get_current_kappa(self) -> float:
        return self.kappa

    def get_total_queries(self) -> int:
        return self.total_queries

    def get_summary(self) -> str:
        if self.keep_history and self.history:
            final_rate = self.history[-1][1]
            final_k = self.history[-1][2]
            min_k = min(h[2] for h in self.history)
            max_k = max(h[2] for h in self.history)
        else:
            final_rate = self.current_rate
            final_k = self.kappa
            min_k = self.min_kappa_seen
            max_k = self.max_kappa_seen
        return (f"Final running rate: {final_rate:.1f} queries/100 | "
                f"Total queries: {self.total_queries} | "
                f"Final kappa: {final_k:.3f} (range: {min_k:.3f}–{max_k:.3f})")
