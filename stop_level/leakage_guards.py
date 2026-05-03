"""
stop_level/leakage_guards.py
=============================================================================
Hard-enforced leakage zones for the two prediction scenarios.

Scenario A — pre-departure:
    No actual delay information from ANY stop on the same service is allowed
    to enter `x`. Lagged history (yesterday's run, last week's same trip) is
    fine because it pre-dates the prediction time.

Scenario B — inflight:
    Earlier stops `1..k-1` of the same run ARE observable. Only the current
    and future stops' delays remain forbidden. The `prev_stop_actual_delay`
    and `cum_actual_delay_so_far` features are *whitelisted* features that
    encode what we are allowed to know.

These sets are imported by `preprocess_stops.py` and by the pytest leakage
tests. Any column listed in the relevant set MUST NOT appear in `X`.
"""
from __future__ import annotations


# Universally banned: the label, its direct sources, and anything that
# encodes the same-stop outcome.
_LABEL_AND_SAMESTOP = {
    "y_stop",            # the label itself
    "delay_min",         # raw float kept only for evaluation slicing, never X
    "delay_minutes",
    "arrival_delay",
    "departure_delay",
    "is_disrupted",      # service-level label proxy
    "y_service",
    "cancelled",
    "max_delay_min",
    "final_delay",
    "_late",             # the boolean derived from delay_min > threshold
}

# Features that encode information from OTHER stops on the same service.
# In scenario A every one of these is forbidden. In scenario B the explicit
# "stops 1..k-1 actual delay" features are allowed and listed in
# `_INFLIGHT_WHITELIST`.
_OTHER_STOP_LEAK = {
    "prev_stop_actual_delay",
    "cum_actual_delay_so_far",
    "max_actual_delay_so_far",
    "headway_to_train_ahead",
}

LEAKY_COLS_A: frozenset[str] = frozenset(_LABEL_AND_SAMESTOP | _OTHER_STOP_LEAK)
LEAKY_COLS_B: frozenset[str] = frozenset(_LABEL_AND_SAMESTOP)
_INFLIGHT_WHITELIST: frozenset[str] = frozenset(_OTHER_STOP_LEAK)


def assert_no_leakage(columns, scenario: str) -> None:
    """
    Raise ValueError if any forbidden column is present in `columns`.

    Parameters
    ----------
    columns : iterable[str]
        The set of column names that will be passed as `X` to a model.
    scenario : {"A", "B"}
    """
    cols = set(columns)
    if scenario == "A":
        leaks = cols & LEAKY_COLS_A
    elif scenario == "B":
        leaks = cols & LEAKY_COLS_B
    else:
        raise ValueError(f"scenario must be 'A' or 'B', got {scenario!r}")
    if leaks:
        raise ValueError(
            f"Leakage detected for scenario {scenario}: {sorted(leaks)}"
        )


def feature_columns(all_columns, scenario: str) -> list[str]:
    """Return the subset of `all_columns` that are legal features for `scenario`."""
    leaky = LEAKY_COLS_A if scenario == "A" else LEAKY_COLS_B
    keep = [c for c in all_columns if c not in leaky and not c.startswith("_meta_")]
    if scenario == "A":
        # Belt-and-suspenders: also strip whitelist features in case the
        # caller produced them but is asking for scenario A.
        keep = [c for c in keep if c not in _INFLIGHT_WHITELIST]
    return keep
