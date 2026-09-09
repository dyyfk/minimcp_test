"""Oracle-utilization decomposition R_b (G0, external-review item 6).

For each pool with cached paired never/always native arms, every query
has L (local correct, never arm) and E (expert correct, always arm).
At an exact budget k = round(b*n):

  random  : expected delivered acc = (1-b)*mean(L) + b*mean(E)
  probe   : escalate the top-k by the never-arm onset score among
            is_info-eligible rows (ranking-only view: exact k, no
            threshold calibration error)
  oracle  : knows (L,E); spends budget on fixable rows (L=0,E=1)
            first, then no-op rows, harmful rows (L=1,E=0) only when
            forced.  Upper bound of ANY router at that budget.

  R_b = (probe - random) / (oracle - random)
      = share of the achievable routing gain the probe captures.

Reported per pool and as the external-5 mean, with paired bootstrap
95% CIs (resample queries; absolute deltas kept alongside because R_b
is unstable when oracle-random is small).  Also the joint (L,E) table
per pool: fixable / harmful / both-right / both-wrong shares -- the
ceiling explanation.

Usage (from interactive_paper/):
  .venv_boot\\Scripts\\python.exe scripts\\45_oracle_utilization.py
Outputs: figures/oracle_utilization.json + .png, printed tables.
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

spec = importlib.util.spec_from_file_location(
    "s40", Path(__file__).with_name("40_threshold_sweep.py"))
s40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s40)

RATES = [.05, .10, .15, .20, .25, .30, .40, .50]
EXT5 = ["striviaqa", "swebq", "sllama", "sdqa", "sreason"]
N_BOOT = 1000
RNG = np.random.default_rng(42)


def arm_pair(pool):
    never, always = s40.load_arm(pool, "never"), s40.load_arm(pool, "always")
    ids = never.index.intersection(always.index)
    never, always = never.loc[ids], always.loc[ids]
    L = never["y"].to_numpy()
    E = always["y"].to_numpy()
    score = never["score"].to_numpy(float)
    info = never["is_info"].fillna(True).astype(bool).to_numpy()
    return L, E, score, info, len(never), len(ids)


def accs(L, E, score, info, b):
    n = len(L)
    k = int(round(b * n))
    rand = (1 - b) * L.mean() + b * E.mean()
    # probe: top-k by score among is_info rows (non-info never fire)
    s = np.where(info, score, -np.inf)
    order = np.argsort(-s, kind="stable")
    fire = np.zeros(n, bool)
    fire[order[:k]] = True
    fire &= info  # if k exceeds the info rows, the rest stay local
    probe = np.where(fire, E, L).mean()
    # oracle: fixable first, then no-ops, harmful only when forced
    n_fix = int(((L == 0) & (E == 1)).sum())
    n_harm = int(((L == 1) & (E == 0)).sum())
    n_zero = n - n_fix - n_harm
    gain = min(k, n_fix) - max(0, k - n_fix - n_zero)
    oracle = L.mean() + gain / n
    return rand, probe, oracle


def pool_curves(L, E, score, info):
    return np.array([accs(L, E, score, info, b) for b in RATES])  # (8,3)


def main():
    data = {p: arm_pair(p) for p in s40.POOLS}
    out = {"budgets": RATES, "n_boot": N_BOOT, "pools": {}}

    boots = {}
    for pool, (L, E, score, info, n_never, n_pair) in data.items():
        cur = pool_curves(L, E, score, info)
        bs = np.empty((N_BOOT, len(RATES), 3))
        for r in range(N_BOOT):
            i = RNG.integers(0, n_pair, n_pair)
            bs[r] = pool_curves(L[i], E[i], score[i], info[i])
        boots[pool] = bs
        j = {"fixable": float(((L == 0) & (E == 1)).mean()),
             "harmful": float(((L == 1) & (E == 0)).mean()),
             "both_right": float(((L == 1) & (E == 1)).mean()),
             "both_wrong": float(((L == 0) & (E == 0)).mean())}
        rows = []
        for bi, b in enumerate(RATES):
            rand, probe, oracle = cur[bi]
            d_pr = bs[:, bi, 1] - bs[:, bi, 0]
            d_or = bs[:, bi, 2] - bs[:, bi, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                r_bs = np.where(np.abs(d_or) > 1e-9, d_pr / d_or, np.nan)
            rows.append({
                "budget": b, "random": round(rand, 4),
                "probe": round(probe, 4), "oracle": round(oracle, 4),
                "delta_probe": round(probe - rand, 4),
                "delta_probe_ci": [round(x, 4) for x in
                                   np.percentile(d_pr, [2.5, 97.5])],
                "delta_oracle": round(oracle - rand, 4),
                "R": round((probe - rand) / (oracle - rand), 3)
                     if oracle - rand > 1e-9 else None,
                "R_ci": [round(x, 3) for x in
                         np.nanpercentile(r_bs, [2.5, 97.5])],
            })
        out["pools"][pool] = {
            "n_pair": n_pair, "n_never": n_never, "joint": j,
            "acc_never": round(float(L.mean()), 4),
            "acc_always": round(float(E.mean()), 4),
            "info_share": round(float(info.mean()), 4),
            "curve": rows}

    # external-5 mean: average per-pool accs, R from the means;
    # CI from per-pool-independent bootstrap of the means
    cur5 = np.mean([pool_curves(*data[p][:4]) for p in EXT5], axis=0)
    bs5 = np.mean([boots[p] for p in EXT5], axis=0)
    rows = []
    for bi, b in enumerate(RATES):
        rand, probe, oracle = cur5[bi]
        d_pr = bs5[:, bi, 1] - bs5[:, bi, 0]
        d_or = bs5[:, bi, 2] - bs5[:, bi, 0]
        r_bs = d_pr / d_or
        rows.append({
            "budget": b, "random": round(rand, 4),
            "probe": round(probe, 4), "oracle": round(oracle, 4),
            "delta_probe": round(probe - rand, 4),
            "delta_probe_ci": [round(x, 4) for x in
                               np.percentile(d_pr, [2.5, 97.5])],
            "delta_oracle": round(oracle - rand, 4),
            "R": round((probe - rand) / (oracle - rand), 3),
            "R_ci": [round(x, 3) for x in
                     np.nanpercentile(r_bs, [2.5, 97.5])],
        })
    out["ext5_mean"] = rows

    for pool in list(s40.POOLS) + ["ext5_mean"]:
        d = out["pools"][pool]["curve"] if pool != "ext5_mean" \
            else out["ext5_mean"]
        hdr = f"=== {pool}"
        if pool != "ext5_mean":
            p = out["pools"][pool]
            hdr += (f" (n={p['n_pair']}, fixable {p['joint']['fixable']:.2f},"
                    f" harmful {p['joint']['harmful']:.2f})")
        print(f"\n{hdr} ===")
        print("budget  random  probe  oracle  d_probe  d_oracle    R [CI]")
        for w in d:
            r = "  n/a" if w["R"] is None else f"{w['R']:5.2f}"
            print(f"{w['budget']*100:4.0f}%  {w['random']:.3f}  "
                  f"{w['probe']:.3f}  {w['oracle']:.3f}  "
                  f"{w['delta_probe']:+.3f}  {w['delta_oracle']:+.3f}  "
                  f"{r} [{w['R_ci'][0]:.2f},{w['R_ci'][1]:.2f}]")

    Path("figures/oracle_utilization.json").write_text(
        json.dumps(out, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for pool in EXT5 + ["frozen"]:
        c = out["pools"][pool]["curve"]
        axes[0].plot(RATES, [w["R"] for w in c], marker="o", ms=3,
                     label=pool, lw=1)
    axes[0].plot(RATES, [w["R"] for w in out["ext5_mean"]], "k-",
                 marker="s", ms=4, lw=2, label="ext-5 mean")
    axes[0].set_xlabel("escalation budget")
    axes[0].set_ylabel("R = (probe-rand)/(oracle-rand)")
    axes[0].set_ylim(-0.05, 1.0)
    axes[0].legend(fontsize=6)
    e = out["ext5_mean"]
    axes[1].plot(RATES, [w["random"] for w in e], label="random", lw=1)
    axes[1].plot(RATES, [w["probe"] for w in e], label="probe", lw=1.5)
    axes[1].plot(RATES, [w["oracle"] for w in e], label="oracle", lw=1,
                 ls="--")
    axes[1].set_xlabel("escalation budget")
    axes[1].set_ylabel("delivered accuracy (ext-5 mean)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/oracle_utilization.png", dpi=160)
    print("\nwrote figures/oracle_utilization.{json,png}")


if __name__ == "__main__":
    main()
