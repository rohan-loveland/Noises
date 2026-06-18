"""
Spectrogram A_RED Runner with Shifting Kappa
Adapts the existing A_RED implementation for streaming .npy spectrograms.
Uses flattened spectrogram vectors as high-dimensional input.
The implementation does NOT know true labels until the Oracle is queried.

NEW: Live "Shifting Kappa" (paranoia) controller adjusts kappa on the fly using a
longer-term average of queries per 100 points (larger sliding window + optional
exponential moving average / EMA). Target defaults to ~4-10 queries / 100 points
(adjustable via CLI --target or the constant). Kappa increase → more queries;
decrease → fewer queries. Soft/trending control to respect limited oracle
response capacity. The longer-term averaging prevents drastic kappa swings from
short-term bursts (e.g. 5 queries in one local section vs 30 in the next).

Live status uses a compact ML-style progress line (frequency controlled by
STATUS_UPDATE_EVERY, default every 100 points) showing overall progress + current
kappa + running (smoothed) query rate + seen/discovered class counts, instead of per-adjustment spam.
A ClassDiscoveryCounter tracks per-class first appearance (in the fed stream) vs.
first actual oracle query + how many of each were seen before first query.
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter, deque

# Add paths for imports (A_RED module and local Spectrogram adapter)
sys.path.insert(0, str(Path(__file__).parent.parent / "A_REDimplementation" / "A_RED"))
sys.path.insert(0, str(Path(__file__).parent))

from SpectrogramDataStream import SpectrogramDataStream, SpectrogramOracle
from A_RED import ARED
from Stats import Stats

# ====================== CONFIG ======================
# A_RED Parameters (tuned for high-dim spectrogram data ~40k dims)
INITIAL_KAPPA = 2.1            # Starting value; will be shifted live by the controller
DATA_WINDOW_SIZE = 250        # Smaller memory bound for spectrogram streaming (faster forgetting of old points)
K_COMP_CLUST = 5               # More clusters for comparison in high-variance audio data
QS_VAR = 0                     # Diameter (stable with our 0.1 floor fix)
REL_PROC_VAR = 0               # Disabled: avoids extra queries on o_pts during relevance_processing for new non-bird classes
VERBOSE_FLAGS = []      # 0=summary, 1=add_o_pt prints (confirms non-query path), 2=anomalous checks

NUM_POINTS_TO_PROCESS = 2000     # Minimal for quick verification run (npy loads + BallTree are heavy); demonstrates algorithm with low queries
N_REL_CLASSES = 5              # Target number of relevant classes to discover (for reporting)

# ====================== SHIFTING KAPPA CONFIG ======================
# Target running query rate (e.g. ~4-10 queries per 100 points). This is the primary
# knob for simulating limited oracle (human labeler) response capacity.
#
# Semantics:
#   - Higher kappa = higher "paranoia" = more points declared anomalous = more queries.
#   - Lower kappa = lower paranoia = trust existing clusters more = fewer queries.
# The controller *decreases* kappa when rate is too high, *increases* when too low.
#
# Longer-term averaging:
#   Larger sliding window (500) + EMA for the rate estimate used in control.
#
# Intelligent (PID-style) adjustment (new):
#   Instead of a fixed tiny step every time, we now use a simple PI controller on the
#   rate error. When the observed query rate is FAR from the target, we make a bigger
#   change to kappa. When it is close, we make very small fine-tuning adjustments.
#   This gives much better convergence behavior than the old fixed-step approach.
TARGET_QUERIES_PER_100 = 4     # Desired long-term running average (queries per 100 points)
KAPPA_WINDOW_SIZE = 500         # Size of the sliding window for the raw "last N" rate
USE_EMA = True                  # Use EMA (in addition to the window) for the control signal
EMA_SPAN = 500                  # EMA memory length
KAPPA_MIN = 0.3
KAPPA_MAX = 8.0
KAPPA_ADJUST_EVERY = 20         # Consider adjustment only every N points (stability)
KAPPA_WARMUP_POINTS = 50        # No adjustments until this many points have been seen

# PID-style adaptive controller parameters
KAPPA_PID_KP = 0.65             # Proportional gain - main driver: bigger error → bigger step
KAPPA_PID_KI = 0.04             # Integral gain - removes steady-state offset over longer periods
KAPPA_PID_KD = 0.0              # Derivative gain (keep 0 unless you want to react to the speed of change)
KAPPA_PID_MAX_STEP = 0.07       # Never change kappa by more than this in a single adjustment
KAPPA_PID_MIN_STEP = 0.003      # Smallest allowed adjustment (fine tuning when close to target)
KAPPA_PID_DEADBAND = 0.012      # If |error| < this (in fraction units), do not adjust at all


# Status / progress output control (to avoid spam / wall of text on long runs)
STATUS_UPDATE_EVERY = 100       # Print the compact progress/status line every N points. Larger = much less output. Final status is always shown.


class ShiftingKappaController:
    """
    Live controller that observes query decisions and mutates ared.kappa
    so that a *longer-term* average of queries per 100 points trends toward
    the configured target.

    Kappa acts as a "paranoia" level:
      - Increase kappa → more queries (higher paranoia).
      - Decrease kappa → fewer queries (lower paranoia).

    The controller uses a simple PI (proportional + integral) controller on the
    rate error to decide *how big* the next kappa change should be:

        error = current_rate_fraction - target_fraction
        delta ≈ -(Kp * error + Ki * integral)

    - When the query rate is FAR from the target (large |error|), a bigger step
      is taken so kappa moves quickly.
    - When the rate is close to target, only very small fine-tuning steps are made.
    - The integral term helps correct persistent offsets over many adjustments.

    Rate estimation still uses the long-term window + optional EMA (see config).
    Adjustments remain infrequent (adjust_every) and are skipped inside a deadband.
    """

    def __init__(self,
                 target_queries_per_100: int = 10,
                 window_size: int = 500,
                 min_kappa: float = 0.3,
                 max_kappa: float = 8.0,
                 adjust_every: int = 20,
                 warmup_points: int = 50,
                 initial_kappa: float = None,
                 verbose: bool = True,
                 use_ema: bool = True,
                 ema_span: int = 500,
                 # PID gains and limits (new intelligent adjustment)
                 kp: float = 0.65,
                 ki: float = 0.04,
                 kd: float = 0.0,
                 max_step: float = 0.07,
                 min_step: float = 0.003,
                 deadband: float = 0.012,
                 keep_history: bool = True):
        self.target = float(target_queries_per_100)
        self.window = deque(maxlen=window_size)
        self.min_k = min_kappa
        self.max_k = max_kappa
        self.adjust_every = adjust_every
        self.warmup_points = warmup_points
        self.verbose = verbose
        self.keep_history = keep_history

        # EMA for rate estimation
        self.use_ema = use_ema
        if use_ema and ema_span > 0:
            self.ema_alpha = 2.0 / (ema_span + 1.0)
        else:
            self.ema_alpha = 0.0
        self.ema_fraction = 0.0

        # PID controller state
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
        self.history = [] if keep_history else None  # full history only if needed (saves mem/allocs on long runs)
        self.min_kappa_seen = self.kappa
        self.max_kappa_seen = self.kappa

    def record_and_adjust(self, ared, did_query: bool):
        """Call after every point. Updates rate estimates and may adjust kappa using PI control."""
        self.window.append(1 if did_query else 0)
        self.points += 1
        if did_query:
            self.total_queries += 1

        # Raw window stats
        wlen = max(1, len(self.window))
        window_fraction = sum(self.window) / wlen
        window_rate = window_fraction * 100.0

        # EMA update
        if self.use_ema:
            is_q = 1.0 if did_query else 0.0
            self.ema_fraction = (1.0 - self.ema_alpha) * self.ema_fraction + self.ema_alpha * is_q

        # Control signal (the smoothed long-term rate we actually steer)
        if self.use_ema:
            control_fraction = self.ema_fraction
            rate_per_100 = self.ema_fraction * 100.0
        else:
            control_fraction = window_fraction
            rate_per_100 = window_rate

        self.current_rate = rate_per_100

        if self.keep_history and self.history is not None:
            self.history.append((self.points, rate_per_100, ared.kappa))

        # Track kappa range for summary even without full history (important for --fast long runs)
        k = ared.kappa
        if k < self.min_kappa_seen:
            self.min_kappa_seen = k
        if k > self.max_kappa_seen:
            self.max_kappa_seen = k

        # PI adjustment (only at the chosen cadence after warmup)
        if (self.points >= self.warmup_points and
                self.points % self.adjust_every == 0):

            target_fraction = self.target / 100.0
            error = control_fraction - target_fraction

            old_kappa = ared.kappa

            if abs(error) > self.deadband:
                # Accumulate integral (with mild anti-windup)
                self.integral += error
                self.integral = max(-8.0, min(8.0, self.integral))

                derivative = error - self.prev_error

                # PI output (we negate because positive error = too many queries → we want lower kappa)
                raw_delta = -(self.kp * error + self.ki * self.integral + self.kd * derivative)

                # Turn the raw output into a sensible step size (bigger error → bigger move, but clamped)
                step_size = abs(raw_delta)
                step_size = max(self.min_step, min(self.max_step, step_size))

                if error > 0:
                    # Rate too high → reduce paranoia (lower kappa)
                    new_kappa = max(self.min_k, old_kappa - step_size)
                else:
                    # Rate too low → increase paranoia (raise kappa)
                    new_kappa = min(self.max_k, old_kappa + step_size)

                ared.kappa = new_kappa

            # Always remember the error for the next derivative term
            self.prev_error = error

            if self.verbose and abs(ared.kappa - old_kappa) > 1e-9:
                direction = "↑" if ared.kappa > old_kappa else "↓"
                print(f"    [ShiftingKappa] paranoia {direction} {old_kappa:.3f} → {ared.kappa:.3f} "
                      f"(rate={rate_per_100:.1f}/100, error={error:.3f}, target={self.target})")

        # keep local view in sync
        self.kappa = ared.kappa

    def get_current_rate(self) -> float:
        return self.current_rate

    def get_current_kappa(self) -> float:
        return self.kappa

    def get_total_queries(self) -> int:
        return self.total_queries

    def get_summary(self):
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
                f"Final kappa: {final_k:.3f} (range during run: {min_k:.3f}–{max_k:.3f})")


class ClassDiscoveryCounter:
    """
    Tracks for each class (label):
      - The processed point index when the class first appeared in the incoming data stream.
      - The processed point index when the class was first actually queried (oracle called for it).
      - How many instances ("how many of them") of the class were seen (via direct label peek
        on *every* processed point) before the first query for that class. These are the points
        that arrived and were typically absorbed as o_pts under other clusters without spending
        an oracle query on the new label. Useful for measuring "missed" examples under a query budget.
    This allows measuring both discovery *latency* (point delay) and *volume seen before discovery*
    (count of same-class points before the query that revealed it).
    Uses the ARED processed-point index space (0,1,2,...) so delays are in "algorithm time".
    """

    def __init__(self):
        self.first_appearance = {}       # label -> first processed_idx
        self.first_queried = {}          # label -> first processed_idx (only when oracle was called)
        self.total_seen = {}             # label -> total # of points of this class seen in the whole run
        self.seen_before_queried = {}    # label -> # of this class seen (on appearance) strictly before first query

    def record_appearance(self, label, processed_idx: int):
        if label not in self.first_appearance:
            self.first_appearance[label] = processed_idx
        self.total_seen[label] = self.total_seen.get(label, 0) + 1

        # Only count toward "before queried" while we have not yet spent a query on this label.
        if label not in self.first_queried:
            self.seen_before_queried[label] = self.seen_before_queried.get(label, 0) + 1

    def record_query(self, label, processed_idx: int):
        if label not in self.first_queried:
            self.first_queried[label] = processed_idx
            # The appearance for the discovering point itself has already incremented seen_before_queried.
            # "How many we see before they are queried" should exclude the point that caused the query.
            pre = self.seen_before_queried.get(label, 0)
            self.seen_before_queried[label] = max(0, pre - 1)

    def get_seen_count(self) -> int:
        return len(self.first_appearance)

    def get_discovered_count(self) -> int:
        return len(self.first_queried)

    def get_total_seen(self, label: str) -> int:
        return self.total_seen.get(label, 0)

    def get_seen_before_queried(self, label: str) -> int:
        return self.seen_before_queried.get(label, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectrogram A_RED with live Shifting Kappa for controlled oracle query rate."
    )
    parser.add_argument(
        "--target", "-t", type=float, default=None,
        help="Target running query rate (queries per 100 points). Default: 10. "
             "This directly controls desired oracle load (soft target, not a hard budget)."
    )
    parser.add_argument(
        "--initial-kappa", type=float, default=None,
        help="Starting kappa value (higher = stricter anomaly test = fewer queries). "
             "Controller will shift it live. Default: 2.1"
    )
    parser.add_argument(
        "--num-points", type=int, default=None,
        help="Number of points to process (-1 for all available). Default: 2000 (quick verification)."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Maximum speed mode for long runs (thousands+ points): disable ClassDiscoveryCounter (no per-point label recording or 'seen/disc' stats) "
             "and skip full kappa history tracking in the controller. Status bar and final reports are abbreviated. "
             "The core ARED + kappa controller still run unchanged."
    )
    # Finer controller params remain source-configurable for now (see KAPPA_* constants).
    # User can still override TARGET etc. without editing for day-to-day rate experiments.
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides (if provided) so the rest of main + controller setup
    # continue to use the same global names and f-strings with no further changes.
    global INITIAL_KAPPA, NUM_POINTS_TO_PROCESS, TARGET_QUERIES_PER_100
    if args.target is not None:
        TARGET_QUERIES_PER_100 = args.target
    if args.initial_kappa is not None:
        INITIAL_KAPPA = args.initial_kappa
    if args.num_points is not None:
        NUM_POINTS_TO_PROCESS = args.num_points

    fast_mode = bool(getattr(args, "fast", False))

    print("=== Spectrogram A_RED with Shifting Kappa ===")
    print("Using .npy spectrograms from 5sSpectrograms_tensors/")
    print(f"Starting kappa={INITIAL_KAPPA}, target={TARGET_QUERIES_PER_100} queries per 100 points (long-term running avg)")
    ema_desc = f" + EMA(span={EMA_SPAN})" if USE_EMA else ""
    fast_note = " [FAST MODE: no discovery stats, no full history]" if fast_mode else ""
    print(f"Controller: window={KAPPA_WINDOW_SIZE}{ema_desc}, PI-adaptive (Kp={KAPPA_PID_KP}, Ki={KAPPA_PID_KI}), "
          f"clamp=[{KAPPA_MIN}, {KAPPA_MAX}], adjust every {KAPPA_ADJUST_EVERY} pts after warmup{fast_note}")
    print()

    # Initialize data stream (loads CSV metadata, streams .npy on demand, flattens to 1D vector)
    # Use paths relative to this script so it works from any CWD and on Linux/Windows
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_stream = SpectrogramDataStream(
        csv_path=str(project_root / "5sSpectrograms_tensors" / "train_5s_spectrograms_linux.csv"),
        tensor_dir=str(project_root / "5sSpectrograms_tensors"),
        max_samples=NUM_POINTS_TO_PROCESS if NUM_POINTS_TO_PROCESS > 0 else None,
        shuffle=True,  # Deterministic order for reproducible test
        seed=69,
        label_column="class_name"   # broad categories (Amphibia, Aves, Insecta, ...)
    )

    # Discovery counter: only created when not in fast mode (saves per-point dict/record overhead for long runs).
    # Must be created before oracle so we can pass it as the tracker (used inside answer_query on real queries).
    discovery_counter = None if fast_mode else ClassDiscoveryCounter()

    # Oracle knows true labels but only reveals on query (simulates human-in-loop).
    # discovery_tracker enables the per-class first_queried timestamps (without affecting query budget logic).
    oracle = SpectrogramOracle(data_stream, discovery_tracker=discovery_counter)

    # Initialize A_RED with the *initial* kappa. The controller will mutate ared.kappa live.
    ared = ARED(
        oracle=oracle,
        kappa=INITIAL_KAPPA,
        data_window_size=DATA_WINDOW_SIZE,
        k_comparison_clusters=K_COMP_CLUST,
        QS_VAR=QS_VAR,
        REL_PROC_VAR=REL_PROC_VAR,
        VERBOSE_FLAGS=VERBOSE_FLAGS
    )

    controller = ShiftingKappaController(
        target_queries_per_100=TARGET_QUERIES_PER_100,
        window_size=KAPPA_WINDOW_SIZE,
        min_kappa=KAPPA_MIN,
        max_kappa=KAPPA_MAX,
        adjust_every=KAPPA_ADJUST_EVERY,
        warmup_points=KAPPA_WARMUP_POINTS,
        initial_kappa=INITIAL_KAPPA,
        verbose=False,   # Set True only if you want a [ShiftingKappa] line on every adjustment
        use_ema=USE_EMA,
        ema_span=EMA_SPAN,
        # New PID-style parameters (adaptive step size based on error magnitude)
        kp=KAPPA_PID_KP,
        ki=KAPPA_PID_KI,
        kd=KAPPA_PID_KD,
        max_step=KAPPA_PID_MAX_STEP,
        min_step=KAPPA_PID_MIN_STEP,
        deadband=KAPPA_PID_DEADBAND,
        keep_history=not fast_mode
    )

    print(f"Initialized A_RED + ShiftingKappaController (initial kappa={INITIAL_KAPPA})")
    if fast_mode:
        print("FAST MODE: discovery tracking and full kappa history disabled for speed. Only core rate/kappa control active.")
    else:
        print("Oracle returns relevance=False for ALL classes (queries occur only on anomalies).")
        print("Kappa (paranoia) is shifted gradually: decrease kappa to lower the running query rate, increase to raise it.")
        print("Live status is shown periodically (no per-adjustment spam by default). Class discovery (seen vs. queried) tracked.")
    print()

    print(f"Starting A_RED on {data_stream.n_samples} spectrogram samples...")
    start_time = time.time()

    # Process first point to initialize
    try:
        first_point = data_stream.stream_new_data_point()
        # Record appearance using *correct source row* (stream_counter-1) so skips (missing .npy) don't mis-map labels.
        # Use processed_idx=0 to match ARED abs_idx space used by oracle for record_query.
        first_source_idx = data_stream.stream_counter - 1
        first_label = data_stream.get_true_label_for_idx(first_source_idx)
        if discovery_counter is not None:
            discovery_counter.record_appearance(first_label, 0)

        ared.process_first_point(first_point)
        print("First point processed. Initial cluster created.")
        initial_cluster = ared.subspace_partition.cluster_list[0]
        print(f"Initial cluster comp_distance: {initial_cluster.comp_distance:.4f}")

        # First point is *always* queried (oracle already called record_query for it at idx 0)
        controller.record_and_adjust(ared, did_query=True)
    except Exception as e:
        print(f"Error on first point: {e}")
        return

    # Process remaining points
    points_processed = 1

    while (NUM_POINTS_TO_PROCESS == -1 or points_processed < NUM_POINTS_TO_PROCESS) and data_stream.get_remaining_num_points() > 0:
        try:
            data_point = data_stream.stream_new_data_point()

            # Record appearance for *every* point using proper source row for label (handles skips)
            # and the ARED processed_idx (current points_processed value == this point's abs_idx in ARED)
            current_idx = points_processed
            source_idx = data_stream.stream_counter - 1
            label = data_stream.get_true_label_for_idx(source_idx)
            if discovery_counter is not None:
                discovery_counter.record_appearance(label, current_idx)

            # Use public oracle count to detect whether this point caused a query (avoids peeking ARED internals every step).
            pre_q = oracle.get_query_count()
            ared.process_point(data_point)
            did_query = oracle.get_query_count() > pre_q
            controller.record_and_adjust(ared, did_query=did_query)

            points_processed += 1

            if points_processed % STATUS_UPDATE_EVERY == 0 or points_processed == NUM_POINTS_TO_PROCESS:
                rate = controller.get_current_rate()
                total_queries = controller.get_total_queries()
                if discovery_counter is not None:
                    n_seen = discovery_counter.get_seen_count()
                    n_disc = discovery_counter.get_discovered_count()
                else:
                    n_seen = n_disc = 0
                total = data_stream.n_samples
                pct = 100.0 * points_processed / max(1, total)
                # Simple text progress bar (ML-training style) + key live metrics.
                # Frequency controlled by STATUS_UPDATE_EVERY (default 100) to keep output volume low.
                bar_len = 18
                filled = int(bar_len * points_processed / max(1, total))
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"[{points_processed:6d}/{total:6d}] |{bar}| {pct:5.1f}% | "
                      f"kappa={ared.kappa:.3f} | runQ/100={rate:.1f} | queries={total_queries} | seen={n_seen} disc={n_disc}")

        except Exception as e:
            print(f"Error at point {points_processed}: {e}")
            break

    total_time = time.time() - start_time
    final_queries = oracle.get_query_count()

    # Stats and Results
    print("\n" + "="*60)
    print("A_RED + SHIFTING KAPPA COMPLETE")
    print("="*60)
    print(f"Points processed: {points_processed:,}")
    print(f"Queries made: {final_queries} ({final_queries/points_processed*100:.2f}% of points)")
    print(f"Known classes discovered: {len(ared.subspace_partition.set_of_known_labels)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Queries per second: {final_queries/total_time:.2f}")

    # Controller summary
    print("\n" + controller.get_summary())

    # Class Discovery Report (first appearance in processed stream vs. first oracle query + volume seen before query)
    # Skipped entirely in --fast mode (the counter was never created).
    if discovery_counter is not None:
        print("\nClass Discovery Report:")
        seen = discovery_counter.get_seen_count()
        disc = discovery_counter.get_discovered_count()
        print(f"  Classes seen in data stream: {seen}")
        print(f"  Classes queried/discovered:  {disc}")
        if seen > 0:
            print("  Per-class (processed_idx; 'seen_before' = # of this class seen on appearance *before* first query):")
            items = []
            for label, appear_idx in discovery_counter.first_appearance.items():
                q_idx = discovery_counter.first_queried.get(label, None)
                delay = (q_idx - appear_idx) if q_idx is not None else None
                items.append((appear_idx, label, q_idx, delay))
            items.sort(key=lambda x: x[0])  # sort by first appearance order
            for appear_idx, label, q_idx, delay in items:
                q_str = str(q_idx) if q_idx is not None else "never"
                d_str = str(delay) if delay is not None else "N/A"
                pre = discovery_counter.get_seen_before_queried(label)
                tot = discovery_counter.get_total_seen(label)
                print(f"    {label}: appeared@{appear_idx}, seen_before={pre} (total={tot}), queried@{q_str}, delay={d_str}")
        else:
            print("  (no classes recorded)")
    else:
        print("\nClass Discovery Report: (disabled via --fast for speed; no per-point tracking was performed)")

    # Cluster summary
    print("\nCluster Summary:")
    for i, cluster in enumerate(ared.subspace_partition.cluster_list):
        if cluster.label is not None:  # valid clusters
            n_l = len(cluster.l_pts)
            n_o = len(cluster.o_pts)
            print(f"  Cluster {i}: label={cluster.label}, relevance={cluster.relevance}, "
                  f"l_pts={n_l}, o_pts={n_o}, comp_dist={cluster.comp_distance:.4f}")

    # Oracle stats
    print(f"\nOracle queries: {oracle.get_query_count()}")

    # Save stats if desired
    stats = Stats(ared)
    print("\nDetailed stats available in Stats object.")

    print("\n--- Shifting Kappa Notes ---")
    print("Kappa (paranoia level) was adjusted live using a longer-term rate estimate (window + EMA) + PI controller.")
    print("The PI controller makes *larger* kappa changes when the query rate is far from target, and tiny fine-tuning steps when close.")
    print("Increase kappa (more paranoia) to get more queries; decrease kappa to reduce the long-term running query rate.")
    print("All controller parameters (target, EMA span, Kp/Ki, max/min step, deadband, etc.) are at the top of the file.")
    print("True labels were ONLY used by the Oracle when A_RED decided to query.")
    print("No classes are marked relevant (Oracle always returns relevance=False).")
    print("Status is shown periodically in a compact progress-bar style (see main loop) to avoid thousands of log lines.")
    print("ClassDiscoveryCounter recorded first appearance + first query time *and* how many of each class were seen before the first query.")


if __name__ == "__main__":
    main()
