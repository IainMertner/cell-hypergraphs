"""The Nadeau-Bengio corrected resampled t-test, on its own.

Lives apart from train_patterns.py so that combine_results.py -- which only
reads result JSON -- can import it without pulling in torch, torch_geometric
and models.py. A reporting script that loads the whole training stack fails
whenever that stack is unhappy for reasons having nothing to do with reporting:
another job holding the GPU, a threading collision, memory pressure. Reading
finished results should not be able to break for any of those reasons.
"""

import numpy as np
from scipy import stats


def corrected_t_test(diffs, n_test, n_train):
    """Nadeau & Bengio corrected resampled t-test on paired per-run differences.

    Repeated k-fold runs share training data, so a plain t-test understates the
    variance and calls differences significant that a rerun would not reproduce.
    The correction inflates it by (1/n + n_test/n_train).

    Returns (mean difference, t, two-sided p, half-width of the 95% interval).

    The interval is what a null result should be reported as: "no difference
    larger than X" says how big an effect the design could have found, where a
    p-value only says one was not found. It is computed from the spread of the
    PAIRED differences, which is smaller than either arm's own spread because
    both arms see identical folds and the fold-to-fold variation cancels --
    using an arm's spread instead inflates it.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan")
    var = float(d.var(ddof=1))
    if var == 0.0:
        return mean, float("nan"), (1.0 if mean == 0.0 else 0.0), 0.0
    se = np.sqrt(var * (1.0 / n + n_test / max(n_train, 1)))
    t = mean / se
    half = float(stats.t.ppf(0.975, df=n - 1) * se)
    return mean, float(t), float(2 * stats.t.sf(abs(t), df=n - 1)), half
