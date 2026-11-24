"""The reported draft field changes scale overnight at 2023-12-31 -> 2024-01-01.

Measured on stationary pings of the filtered fleet:

    era        rows      % null   q1    median   q3    IQR
    2023       137.1 M   8.28     8.2   10.1     12.6  4.4
    2024-2026  342.1 M   1.93     7.1    8.4     10.5  3.4

The break is a single overnight step (2023-12-31 median 9.9 and 9.7% null;
2024-01-01 median 8.5 and 1.4% null), and it is not the fleet: of 3,341 vessels
present on both sides with more than 500 stationary pings each, the median
draft ratio is 0.926 and only 29% are within 5% of unchanged.  A real change of
loading could not be simultaneous, could not move the null rate from 8.6% to
1.5% on the same night, and would not compress the spread.  It is a change in
the source feed at the year boundary.

The per-vessel ratios are not one constant: 29% of vessels are unchanged and the
rest are mostly reduced.  That looks less like a rescale than like a change of
WHICH draft is reported -- a static or maximum draft in 2023, the current
laden draft afterwards -- which would leave the always-full ships unchanged and
pull everyone else down.  So the correction below is a level fix, not a repair.

Left alone the break puts a hard level shift in the MIDDLE of the training
period, so 2023 is mapped linearly onto the 2024-2026 scale, matching median and
IQR.  The map is documented and exactly invertible; 2024 onward is untouched.
It cannot make the two eras measure the same thing, so every draft-derived
feature carries more uncertainty in 2023 than after it, and the residual shows
up in draft_change_share (a 26% step remains after the fix, from the reporting
rate: 8.3% of 2023 pings carry no draft at all against 1.9% later).
"""
ERA_BREAK = "2024-01-01"
Q23, M23 = 4.4, 10.1      # IQR and median, 2023
Q24, M24 = 3.4, 8.4       # IQR and median, 2024-2026
SCALE = Q24/Q23           # 0.7727
def sql(col="draft", day="day"):
    return (f"CASE WHEN {day} < DATE '{ERA_BREAK}' "
            f"THEN {SCALE:.6f}*({col} - {M23}) + {M24} ELSE {col} END")
def apply(d, ts):
    import numpy as np, pandas as pd
    old = pd.to_datetime(ts) < pd.Timestamp(ERA_BREAK)
    return np.where(old, SCALE*(d - M23) + M24, d)
