import numpy as np


def scurve_profile(t, q0, q1, tj, ta, tv, td, vlim, alim, jmax):
    # Equations 3.30a - 3.30g of
    # [1] Trajectory Planning for Automatic Machines and Robots, Luigi Biagiotti, Claudio Melchiorri, 2008.
    tfinal = ta + tv + td
    if t < 0 or t > tfinal:
        raise ValueError(f"t {tfinal} should be in bounds [0, {tfinal}]")

    v0 = v1 = 0
    # acceleration
    if t <= tj:
        q = q0 + jmax / 6 * t ** 3
        v = v0 + jmax / 2 * t ** 2
        a = jmax * t
        j = jmax
    elif t <= ta - tj:
        q = q0 + v_0 * t + alim / 6 * (3 * t ** 2 - 3 * tj * t + tj ** 2)
        v = v_0 + alim * (t - tj / 2)
        a = alim
        j = 0
    elif t <= ta:
        q = q0 + (vlim + v0) * ta / 2 - vlim * (ta - t) + jmax / 6 * (ta - t) ** 3
        v = vlim - jmax / 2 * (ta - t) ** 2
        a = jmax * (ta - t)
        j = -jmax
    # constant speed
    elif tv > 0 and t <= ta + tv:
        q = q0 + (vlim + v0) * ta / 2 + vlim * (t - ta)
        v = vlim
        a = 0
        j = 0
    # deceleration
    elif t <= tfinal - td + tj:
        q = (
            q1
            - (vlim + v1) * td / 2
            + vlim * (t - tfinal + td)
            - jmax / 6 * (t - tfinal + td) ** 3
        )
        v = vlim - jmax / 2 * (t - tfinal + td) ** 2
        a = -jmax * (t - tfinal + td)
        j = -jmax
    elif t <= tfinal - tj:
        q = (
            q1
            - (vlim + v1) * td / 2
            + vlim * (t - tfinal + td)
            - alim
            / 6
            * (3 * (t - tfinal + td) ** 2 - 3 * tj * (t - tfinal + td) + tj ** 2)
        )
        v = vlim - alim * (t - tfinal + td - tj / 2)
        a = -jmax * tj
        j = 0
    elif t <= tfinal:
        q = q1 - v1 * (tfinal - t) - jmax / 6 * (tfinal - t) ** 3
        v = v1 + jmax / 2 * (tfinal - t) ** 2
        a = -jmax * (tfinal - t)
        j = jmax
    prof = np.hstack((q, v, a, j))
    return prof


def create_scurve(q0, q1, vmax, amax, jmax):
    # assumes null initial and final velocities as explained in 3.4.3 of [1]
    h = q1 - q0

    if vmax * jmax >= amax ** 2:
        tj = amax / jmax
        ta = tj + vmax / amax
    else:
        tj = np.sqrt(vmax / jmax)
        ta = 2 * tj
    tv = h / vmax - ta
    if tv <= 0:
        tv = 0
        if h >= 2 * amax ** 3 / jmax ** 2:
            tj = amax / jmax
            ta = tj / 2 + np.sqrt((tj / 2) ** 2 + h / amax)
        else:
            tj = np.power(h / (2 * jmax), 1 / 3)
            ta = 2 * tj
    td = ta
    alim = jmax * tj
    vlim = alim * (ta - tj)
    tfinal = ta + tv + td
    scurve_fn = lambda t: scurve_profile(t, q0, q1, tj, ta, tv, td, vlim, alim, jmax)

    return scurve_fn, tfinal
