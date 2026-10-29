#!/usr/bin/env python3
"""
SportLens - model v2 experiments (E1..E7)  -  runs on your Mac, reads/writes Supabase.

What it does
------------
Rolling-origin backtest (refit every N days, only past data, time-decay weights) of a
Poisson goal model and variants, compared with de-margined closing odds (Avg and Pinnacle).

  E1  time-decay half-life 90 / 180 / 365   (+ ridge low/high as sanity)
  E2  Dixon-Coles low-score correction
  E3  joint per-country model (both tiers in one fit)
  E4  shots-based pseudo-xG target (30% / 60% blend)
  E5  Elo as extra feature (logistic stacking)
  E6  rest days as extra feature (logistic stacking)
  E7  calibration + blend with market PRE-match odds (never closing -> no leakage)
  +   'combo' = automatic combination of the changes that passed on the DEV period

Honesty rules built in
----------------------
* no look-ahead: every fit uses matches strictly before the refit date; stacking models
  are trained on earlier out-of-sample predictions only (expanding monthly window);
* DEV period = eval_from .. dev_to (default 2023-08-01 .. 2025-06-30);
  HOLD-OUT = 2025-07-01 onward, only evaluated with --holdout. Run the hold-out ONCE,
  with the config chosen on DEV (use --combo-spec), otherwise it is no longer a hold-out;
* a change "passes" only if pooled log-loss improves vs base in BOTH tiers AND at least
  4 of 5 leagues of each tier improve;  significance (paired bootstrap CI) is reported;
* the market is the yardstick: diff_vs_market > 0 means the model is WORSE than the market.

Setup (once)
------------
  pip install pandas numpy scipy scikit-learn "psycopg[binary]"
  export DATABASE_URL='postgresql://postgres:...@db.xxxx.supabase.co:5432/postgres'   # in YOUR terminal

Run
---
  python experiments_v2.py --source synthetic --step-days 28 --bootstrap 200      # quick self-test
  python experiments_v2.py --batch dev1 --jobs 4                                   # DEV experiments
  python experiments_v2.py --batch hold1 --holdout --runs base,combo \
         --combo-spec '{"hl":180,"dc":true,"joint":false,"xg_w":0,"ridge":0.3,"feats":["elo"]}'
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize, nnls
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

MAXG = 8
G = np.arange(MAXG + 1)
TRI_H = (G[:, None] > G[None, :]).astype(float)
TRI_D = np.eye(MAXG + 1)
TRI_A = (G[:, None] < G[None, :]).astype(float)
OVER = ((G[:, None] + G[None, :]) > 2).astype(float)
YMAP = {"H": 0, "D": 1, "A": 2}

BASE_CFG = {"hl": 180, "dc": False, "joint": False, "xg_w": 0.0, "ridge": 0.3}


# =============================================================== data loading
def q(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def load_db(url, live=False):
    """live=True also returns not-yet-played fixtures dated today or later (hg/ag = NaN)."""
    import psycopg
    played = "(m.home_goals is not null and m.away_goals is not null)"
    cond = f"({played} or m.match_date >= current_date)" if live else played
    with psycopg.connect(url) as conn:
        m = q(conn, f"""
            select m.id as mid, l.code as league, l.country, l.tier, m.season, m.match_date,
                   m.home_team_id as home_id, m.away_team_id as away_id,
                   m.home_goals as hg, m.away_goals as ag, m.result, m.stats
            from matches m join leagues l on l.id = m.league_id
            where {cond} and l.sport = 'football'
            order by m.match_date, m.id""")
        o = q(conn, """select match_id, bookmaker, market, snapshot, odds
                       from odds_snapshots where bookmaker in ('Avg','PS')""")
    rows = {}
    for mid, bk, mk, sn, od in zip(o.match_id, o.bookmaker, o.market, o.snapshot, o.odds):
        if isinstance(od, str):
            od = json.loads(od)
        r = rows.setdefault(mid, {})
        for k, v in od.items():
            r[f"{bk}|{sn}|{mk}|{k}"] = float(v)
    odf = pd.DataFrame.from_dict(rows, orient="index")
    odf.index.name = "mid"
    df = m.merge(odf, left_on="mid", right_index=True, how="left")
    for key, col in [("home_shots", "hs"), ("away_shots", "as_"),
                     ("home_shots_on_target", "hst"), ("away_shots_on_target", "ast")]:
        df[col] = [float((s or {}).get(key, np.nan)) for s in df["stats"]]
    df = df.drop(columns=["stats"])
    df["match_date"] = pd.to_datetime(df["match_date"])
    if live:
        df["hg"] = df["hg"].astype(float)
        df["ag"] = df["ag"].astype(float)
    else:
        df["hg"] = df["hg"].astype(int)
        df["ag"] = df["ag"].astype(int)
    df["tier"] = df["tier"].astype(int)
    return add_market(df)


def add_market(df):
    """de-margined (proportional) probabilities: mc = Avg closing, pc = Pinnacle closing,
    mp = Avg pre-match;  ao_* = raw Avg pre odds;  ouc_O / oup_O = Avg over-2.5 prob."""
    def tri(bk, sn, pref, raw_pref=None):
        cols = [f"{bk}|{sn}|1X2|{k}" for k in "HDA"]
        if not all(c in df.columns for c in cols):
            for k in "HDA":
                df[f"{pref}_{k}"] = np.nan
                if raw_pref:
                    df[f"{raw_pref}_{k}"] = np.nan
            return
        raw = df[cols].to_numpy(float)
        ok = np.isfinite(raw).all(1) & (raw > 1).all(1)
        with np.errstate(all="ignore"):
            inv = 1.0 / raw
            p = inv / inv.sum(1, keepdims=True)
        for i, k in enumerate("HDA"):
            df[f"{pref}_{k}"] = np.where(ok, p[:, i], np.nan)
            if raw_pref:
                df[f"{raw_pref}_{k}"] = np.where(ok, raw[:, i], np.nan)

    def ou(bk, sn, name):
        co, cu = f"{bk}|{sn}|OU2.5|O", f"{bk}|{sn}|OU2.5|U"
        if co not in df.columns or cu not in df.columns:
            df[name] = np.nan
            return
        o_, u_ = df[co].to_numpy(float), df[cu].to_numpy(float)
        with np.errstate(all="ignore"):
            p = (1 / o_) / (1 / o_ + 1 / u_)
        df[name] = np.where(np.isfinite(p) & (o_ > 1) & (u_ > 1), p, np.nan)

    tri("Avg", "closing", "mc")
    tri("PS", "closing", "pc")
    tri("Avg", "pre", "mp", "ao")
    ou("Avg", "closing", "ouc_O")
    ou("Avg", "pre", "oup_O")
    return df


# ------------------------------------------------------------ synthetic data
def make_synthetic(seed=7, seasons=7, n_teams=20):
    rng = np.random.default_rng(seed)
    rows, mid = [], 0
    countries = ["ENG", "ESP", "GER", "ITA", "FRA"]
    for ci, c in enumerate(countries):
        strength = {}
        for tier in (1, 2):
            for t in range(n_teams):
                tid = ci * 1000 + tier * 100 + t
                strength[tid] = (rng.normal(0.0, 0.30), rng.normal(0.0, 0.30))
        tier_of = {tid: (1 if (tid // 100) % 10 == 1 else 2) for tid in strength}
        for s in range(seasons):
            season = f"{2019 + s}/{str(2020 + s)[2:]}"
            start = pd.Timestamp(f"{2019 + s}-08-10")
            for tier in (1, 2):
                teams = [t for t in strength if tier_of[t] == tier]
                fx = [(a, b) for a in teams for b in teams if a != b]
                rng.shuffle(fx)
                days = np.linspace(0, 280, len(fx))
                for (h, a), dd in zip(fx, days):
                    ah, dh = strength[h]
                    aa, da = strength[a]
                    off = -0.25 if tier == 2 else 0.0
                    lh = math.exp(0.10 + off + 0.30 + ah - da)
                    la = math.exp(0.10 + off + aa - dh)
                    hg, ag = rng.poisson(lh), rng.poisson(la)
                    g = np.arange(9)
                    ph, pa = poisson.pmf(g, lh), poisson.pmf(g, la)
                    M = np.outer(ph, pa)
                    M /= M.sum()
                    true = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
                    pover = (M * OVER).sum()
                    noise = lambda sd: np.exp(rng.normal(0, sd, 3))
                    close = true * noise(0.03)
                    close /= close.sum()
                    pre = true * noise(0.05)
                    pre /= pre.sum()
                    marg = 1.05
                    shots_h = max(2, rng.poisson(10 + 4 * (lh - 1.3)))
                    shots_a = max(2, rng.poisson(10 + 4 * (la - 1.1)))
                    rows.append(dict(
                        mid=mid, league=f"{c}{tier}", country=c, tier=tier, season=season,
                        match_date=start + pd.Timedelta(days=float(dd)),
                        home_id=h, away_id=a, hg=int(hg), ag=int(ag),
                        result="H" if hg > ag else ("D" if hg == ag else "A"),
                        hs=float(shots_h), as_=float(shots_a),
                        hst=float(rng.binomial(shots_h, 0.33)), ast=float(rng.binomial(shots_a, 0.33)),
                        mc_H=close[0], mc_D=close[1], mc_A=close[2],
                        pc_H=close[0], pc_D=close[1], pc_A=close[2],
                        mp_H=pre[0], mp_D=pre[1], mp_A=pre[2],
                        ao_H=marg / pre[0], ao_D=marg / pre[1], ao_A=marg / pre[2],
                        ouc_O=min(max(pover * math.exp(rng.normal(0, .03)), .05), .95),
                        oup_O=min(max(pover * math.exp(rng.normal(0, .05)), .05), .95)))
                    mid += 1
            # promotion / relegation: swap 2 weakest of tier1 with 2 strongest of tier2 (by attack-def)
            t1 = sorted([t for t in strength if tier_of[t] == 1], key=lambda t: sum(strength[t]))
            t2 = sorted([t for t in strength if tier_of[t] == 2], key=lambda t: -sum(strength[t]))
            for a_, b_ in zip(t1[:2], t2[:2]):
                tier_of[a_], tier_of[b_] = 2, 1
    df = pd.DataFrame(rows).sort_values(["match_date", "mid"]).reset_index(drop=True)
    return df


# =============================================================== features
def add_elo(df, k=20.0, ha=60.0):
    df = df.sort_values(["match_date", "mid"]).reset_index(drop=True)
    R, elo_h, elo_a = {}, np.zeros(len(df)), np.zeros(len(df))
    for i, (hid, aid, s, tier, hg, ag) in enumerate(
            zip(df.home_id, df.away_id, df.season, df.tier, df.hg, df.ag)):
        def get(t):
            r, se = R.get(t, (1500.0 if tier == 1 else 1400.0, s))
            return (0.75 * r + 0.25 * 1500.0) if se != s else r
        rh, ra = get(hid), get(aid)
        elo_h[i], elo_a[i] = rh, ra
        if hg != hg or ag != ag:  # unplayed fixture: record pre-match rating, do not update
            continue
        exp = 1.0 / (1.0 + 10 ** (-(rh + ha - ra) / 400.0))
        res = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        delta = k * mult * (res - exp)
        R[hid], R[aid] = (rh + delta, s), (ra - delta, s)
    df["elo_diff"] = elo_h + ha - elo_a
    return df


def add_rest(df):
    a = df[["mid", "match_date", "home_id"]].rename(columns={"home_id": "team"})
    a["side"] = "h"
    b = df[["mid", "match_date", "away_id"]].rename(columns={"away_id": "team"})
    b["side"] = "a"
    long = pd.concat([a, b]).sort_values(["team", "match_date"])
    long["rest"] = long.groupby("team")["match_date"].diff().dt.days
    long["rest"] = long["rest"].fillna(7).clip(2, 14)
    rh = long[long.side == "h"].set_index("mid")["rest"]
    ra = long[long.side == "a"].set_index("mid")["rest"]
    df["rest_diff"] = df["mid"].map(rh).values - df["mid"].map(ra).values
    return df


# =============================================================== Poisson model
def targets(tr, xg_w):
    hg, ag = tr.hg.to_numpy(float), tr.ag.to_numpy(float)
    if xg_w <= 0:
        return hg, ag
    have = tr[["hs", "as_", "hst", "ast"]].notna().all(axis=1).to_numpy()
    if have.sum() < 200:
        return hg, ag
    sub = tr[have]
    X = np.vstack([np.column_stack([sub.hst, sub.hs - sub.hst]),
                   np.column_stack([sub.ast, sub.as_ - sub.ast])])
    y = np.concatenate([sub.hg, sub.ag]).astype(float)
    coef, _ = nnls(X, y)
    xh = (coef[0] * tr.hst + coef[1] * (tr.hs - tr.hst)).to_numpy(float)
    xa = (coef[0] * tr.ast + coef[1] * (tr.as_ - tr.ast)).to_numpy(float)
    yh = np.where(have, (1 - xg_w) * hg + xg_w * xh, hg)
    ya = np.where(have, (1 - xg_w) * ag + xg_w * xa, ag)
    return yh, ya


def fit_poisson(tr, cfg, ref_date):
    teams = pd.unique(np.concatenate([tr.home_id.to_numpy(), tr.away_id.to_numpy()]))
    tidx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    leagues = sorted(tr.league.unique())
    lidx = {x: i for i, x in enumerate(leagues)}
    L = len(leagues)
    h = tr.home_id.map(tidx).to_numpy()
    a = tr.away_id.map(tidx).to_numpy()
    l = tr.league.map(lidx).to_numpy()
    age = (ref_date - tr.match_date).dt.days.to_numpy(float)
    w = 0.5 ** (age / cfg["hl"])
    yh, ya = targets(tr, cfg["xg_w"])
    ridge = cfg["ridge"]

    def f(p):
        att, dfn = p[:n], p[n:2 * n]
        mu, hm = p[2 * n:2 * n + L], p[2 * n + L:]
        eh = mu[l] + hm[l] + att[h] - dfn[a]
        ea = mu[l] + att[a] - dfn[h]
        lh, la = np.exp(eh), np.exp(ea)
        nll = -(w * (yh * eh - lh + ya * ea - la)).sum() + ridge * ((att ** 2).sum() + (dfn ** 2).sum())
        gh, ga = -w * (yh - lh), -w * (ya - la)
        g = np.concatenate([
            np.bincount(h, gh, n) + np.bincount(a, ga, n) + 2 * ridge * att,
            -np.bincount(a, gh, n) - np.bincount(h, ga, n) + 2 * ridge * dfn,
            np.bincount(l, gh + ga, L), np.bincount(l, gh, L)])
        return nll, g

    x0 = np.zeros(2 * n + 2 * L)
    x0[2 * n:2 * n + L] = math.log(max(0.5 * (tr.hg.mean() + tr.ag.mean()), 0.3))
    x0[2 * n + L:] = 0.25
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    p = res.x
    mdl = {"tidx": tidx, "att": p[:n], "def": p[n:2 * n],
           "mu": {x: p[2 * n + i] for x, i in lidx.items()},
           "hm": {x: p[2 * n + L + i] for x, i in lidx.items()}, "rho": 0.0}
    if cfg["dc"]:
        lh = np.exp(p[2 * n:2 * n + L][l] + p[2 * n + L:][l] + p[:n][h] - p[n:2 * n][a])
        la = np.exp(p[2 * n:2 * n + L][l] + p[:n][a] - p[n:2 * n][h])
        hgi, agi = tr.hg.to_numpy(), tr.ag.to_numpy()
        m00, m01 = (hgi == 0) & (agi == 0), (hgi == 0) & (agi == 1)
        m10, m11 = (hgi == 1) & (agi == 0), (hgi == 1) & (agi == 1)
        best, bl = 0.0, -1e18
        for rho in np.linspace(-0.30, 0.10, 41):
            t00, t01 = 1 - lh * la * rho, 1 + lh * rho
            t10, t11 = 1 + la * rho, np.full_like(lh, 1 - rho)
            if min(t00[m00].min() if m00.any() else 1, t01[m01].min() if m01.any() else 1,
                   t10[m10].min() if m10.any() else 1, t11.min()) <= 0:
                continue
            ll = ((w[m00] * np.log(t00[m00])).sum() + (w[m01] * np.log(t01[m01])).sum()
                  + (w[m10] * np.log(t10[m10])).sum() + (w[m11] * np.log(t11[m11])).sum())
            if ll > bl:
                best, bl = rho, ll
        mdl["rho"] = best
    return mdl


def score_probs(lh, la, rho):
    ph = poisson.pmf(G[None, :], lh[:, None])
    pa = poisson.pmf(G[None, :], la[:, None])
    M = ph[:, :, None] * pa[:, None, :]
    if rho != 0.0:
        M[:, 0, 0] *= 1 - lh * la * rho
        M[:, 0, 1] *= 1 + lh * rho
        M[:, 1, 0] *= 1 + la * rho
        M[:, 1, 1] *= 1 - rho
    M /= M.sum(axis=(1, 2), keepdims=True)
    return (np.einsum("nij,ij->n", M, TRI_H), np.einsum("nij,ij->n", M, TRI_D),
            np.einsum("nij,ij->n", M, TRI_A), np.einsum("nij,ij->n", M, OVER))


def run_group(args):
    gdf, cfg, dates, step, train_days = args
    out = []
    for t in dates:
        te = gdf[(gdf.match_date >= t) & (gdf.match_date < t + pd.Timedelta(days=step))]
        if te.empty:
            continue
        tr = gdf[(gdf.match_date < t) & (gdf.match_date >= t - pd.Timedelta(days=train_days))]
        if len(tr) < 150:
            continue
        mdl = fit_poisson(tr, cfg, t)
        te = te[te.league.isin(list(mdl["mu"].keys()))]
        if te.empty:
            continue
        ah = te.home_id.map(lambda x: mdl["att"][mdl["tidx"][x]] if x in mdl["tidx"] else 0.0).to_numpy()
        dh = te.home_id.map(lambda x: mdl["def"][mdl["tidx"][x]] if x in mdl["tidx"] else 0.0).to_numpy()
        aa = te.away_id.map(lambda x: mdl["att"][mdl["tidx"][x]] if x in mdl["tidx"] else 0.0).to_numpy()
        da = te.away_id.map(lambda x: mdl["def"][mdl["tidx"][x]] if x in mdl["tidx"] else 0.0).to_numpy()
        mu = te.league.map(mdl["mu"]).to_numpy()
        hm = te.league.map(mdl["hm"]).to_numpy()
        lh, la = np.exp(mu + hm + ah - da), np.exp(mu + aa - dh)
        pH, pD, pA, pO = score_probs(lh, la, mdl["rho"])
        out.append(pd.DataFrame({"mid": te.mid.to_numpy(), "pH": pH, "pD": pD, "pA": pA, "pO": pO}))
    return pd.concat(out) if out else pd.DataFrame(columns=["mid", "pH", "pD", "pA", "pO"])


def run_base_model(df, cfg, pred_from, pred_to, step, train_years, jobs):
    key = "country" if cfg["joint"] else "league"
    dates = list(pd.date_range(pred_from, pred_to, freq=f"{step}D"))
    tasks = [(g.sort_values("match_date"), cfg, dates, step, int(train_years * 365))
             for _, g in df.groupby(key)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            parts = list(ex.map(run_group, tasks))
    else:
        parts = [run_group(t) for t in tasks]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["mid", "pH", "pD", "pA", "pO"])


# =============================================================== stacking
def feat_matrix(d, feats):
    X = [np.log(d.pH / d.pD), np.log(d.pA / d.pD)]
    if "elo" in feats:
        X.append(d.elo_diff / 400.0)
    if "rest" in feats:
        X.append(d.rest_diff / 7.0)
    if "mkt" in feats:
        X += [np.log(d.mp_H / d.mp_D), np.log(d.mp_A / d.mp_D)]
    return np.column_stack(X)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack(pred, meta, feats, min_n=600, C=10.0):
    cols = ["mid", "match_date", "tier", "result", "elo_diff", "rest_diff",
            "mp_H", "mp_D", "mp_A", "oup_O", "tot"]
    d = pred.merge(meta[cols], on="mid").sort_values("match_date").reset_index(drop=True)
    X = feat_matrix(d, feats)
    ok = np.isfinite(X).all(1) & np.isfinite(d[["pH", "pD", "pA"]].to_numpy()).all(1)
    d, X = d[ok].reset_index(drop=True), X[ok]
    y = d.result.to_numpy()
    month = d.match_date.dt.to_period("M")
    out = np.full((len(d), 4), np.nan)
    out[:, 3] = d.pO.to_numpy()
    for tier in (1, 2):
        tmask = (d.tier == tier).to_numpy()
        for mth in sorted(month[tmask].unique()):
            start = mth.start_time
            trm = tmask & (d.match_date < start).to_numpy()
            tem = tmask & (month == mth).to_numpy()
            if trm.sum() < min_n or not tem.any():
                continue
            clf = LogisticRegression(C=C, max_iter=500).fit(X[trm], y[trm])
            pr = clf.predict_proba(X[tem])
            ci = {c: i for i, c in enumerate(clf.classes_)}
            out[tem, 0], out[tem, 1], out[tem, 2] = pr[:, ci["H"]], pr[:, ci["D"]], pr[:, ci["A"]]
            # over/under: calibrate (and blend with market pre if requested)
            yo = (d.tot.to_numpy()[trm] > 2.5).astype(int)
            xo_tr = [logit(d.pO[trm].to_numpy())]
            xo_te = [logit(d.pO[tem].to_numpy())]
            if "mkt" in feats:
                xo_tr.append(logit(d.oup_O[trm].to_numpy()))
                xo_te.append(logit(d.oup_O[tem].to_numpy()))
            xo_tr, xo_te = np.column_stack(xo_tr), np.column_stack(xo_te)
            okt = np.isfinite(xo_tr).all(1)
            if okt.sum() > 200 and len(np.unique(yo[okt])) == 2:
                co = LogisticRegression(C=C, max_iter=500).fit(xo_tr[okt], yo[okt])
                po = co.predict_proba(np.nan_to_num(xo_te, nan=0.0))[:, 1]
                oke = np.isfinite(xo_te).all(1)
                tem_idx = np.where(tem)[0]
                out[tem_idx[oke], 3] = po[oke]
    res = pd.DataFrame({"mid": d.mid, "pH": out[:, 0], "pD": out[:, 1], "pA": out[:, 2], "pO": out[:, 3]})
    return res.dropna(subset=["pH"]).reset_index(drop=True)


# =============================================================== evaluation
def per_match(P, y):
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
    br = ((P - np.eye(3)[y]) ** 2).sum(1)
    return ll, br


def boot_ci(d, B, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 30:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(math.ceil(B / 100)):
        idx = rng.integers(0, len(d), (100, len(d)))
        means.append(d[idx].mean(1))
    m = np.concatenate(means)[:B]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def ece(P, y, bins=10):
    conf = P.ravel()
    hit = np.eye(3)[y].ravel()
    b = np.minimum((conf * bins).astype(int), bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(tot)


def eval_frame(pred, meta):
    ev = pred.merge(meta, on="mid")
    ev = ev[ev.mc_H.notna()].copy()
    ev["y"] = ev.result.map(YMAP).astype(int)
    P = ev[["pH", "pD", "pA"]].to_numpy()
    ev["ll"], ev["br"] = per_match(P, ev.y.to_numpy())
    M = ev[["mc_H", "mc_D", "mc_A"]].to_numpy()
    ev["ll_mc"], ev["br_mc"] = per_match(M, ev.y.to_numpy())
    ps = ev[["pc_H", "pc_D", "pc_A"]].to_numpy()
    okp = np.isfinite(ps).all(1)
    ev["ll_pc"] = np.nan
    if okp.any():
        ev.loc[okp, "ll_pc"] = per_match(ps[okp], ev.y.to_numpy()[okp])[0]
    yo = (ev.tot > 2.5).astype(float).to_numpy()
    po = np.clip(ev.pO.to_numpy(), 1e-6, 1 - 1e-6)
    ev["ll_ou"] = -(yo * np.log(po) + (1 - yo) * np.log(1 - po))
    pm = np.clip(ev.ouc_O.to_numpy(), 1e-6, 1 - 1e-6)
    ev["ll_ou_mc"] = -(yo * np.log(pm) + (1 - yo) * np.log(1 - pm))
    return ev


def clv_stats(ev, edge, B):
    P = ev[["pH", "pD", "pA"]].to_numpy()
    O = ev[["ao_H", "ao_D", "ao_A"]].to_numpy()
    C = ev[["mc_H", "mc_D", "mc_A"]].to_numpy()
    ok = np.isfinite(O).all(1)
    P, O, C, y = P[ok], O[ok], C[ok], ev.y.to_numpy()[ok]
    if len(y) == 0:
        return {}
    e = P * O - 1
    pick = e.argmax(1)
    take = e[np.arange(len(y)), pick] > edge
    if take.sum() < 30:
        return {"clv_n": int(take.sum())}
    o = O[np.arange(len(y)), pick][take]
    c = C[np.arange(len(y)), pick][take]
    win = (pick[take] == y[take]).astype(float)
    clv = o * c - 1
    roi = o * win - 1
    lo, hi = boot_ci(clv, B)
    rlo, rhi = boot_ci(roi, B)
    return {"clv_n": int(take.sum()), "clv_mean": float(clv.mean()), "clv_lo": lo, "clv_hi": hi,
            "roi_mean": float(roi.mean()), "roi_lo": rlo, "roi_hi": rhi}


def evaluate(name, pred, meta, period_rng, base_ll, B, edge, is_base):
    ev = eval_frame(pred, meta)
    lo, hi = period_rng
    ev = ev[(ev.match_date >= lo) & (ev.match_date <= hi)]
    ev = ev.copy()
    ev["ll_base"] = ev.mid.map(base_ll)
    rows = []
    leagues = sorted(ev.league.unique())
    scopes = [("all", None, ev), ("tier1", None, ev[ev.tier == 1]), ("tier2", None, ev[ev.tier == 2])]
    scopes += [("league", lg, ev[ev.league == lg]) for lg in leagues]
    tier_of = ev.groupby("league").tier.first().to_dict()
    per_league_improved = {}
    for scope, lg, s in scopes:
        if len(s) == 0:
            continue
        m = {"n": int(len(s)), "ll": float(s.ll.mean()), "brier": float(s.br.mean()),
             "ll_market": float(s.ll_mc.mean()), "ece": ece(s[["pH", "pD", "pA"]].to_numpy(), s.y.to_numpy())}
        d = (s.ll - s.ll_mc).to_numpy()
        m["d_vs_market"] = float(d.mean())
        m["d_vs_market_lo"], m["d_vs_market_hi"] = boot_ci(d, B) if scope != "league" else (np.nan, np.nan)
        sp = s[s.ll_pc.notna()]
        if len(sp) > 30:
            m["d_vs_pinnacle"] = float((sp.ll - sp.ll_pc).mean())
            m["n_pinnacle"] = int(len(sp))
        so = s[s.ouc_O.notna() & s.pO.notna()]
        if len(so) > 30:
            m["ou_ll"], m["ou_ll_market"] = float(so.ll_ou.mean()), float(so.ll_ou_mc.mean())
            m["ou_d_vs_market"] = float((so.ll_ou - so.ll_ou_mc).mean())
        if not is_base:
            sb = s[s.ll_base.notna()]
            if len(sb) > 30:
                db = (sb.ll - sb.ll_base).to_numpy()
                m["n_vs_base"] = int(len(sb))
                m["d_vs_base"] = float(db.mean())
                if scope != "league":
                    m["d_vs_base_lo"], m["d_vs_base_hi"] = boot_ci(db, B)
                else:
                    per_league_improved[lg] = bool(db.mean() < 0)
        if scope in ("all", "tier1", "tier2"):
            m.update(clv_stats(s, edge, B))
        rows.append({"scope": scope, "league": lg, "metrics": m})
    if not is_base:
        for r in rows:
            if r["scope"] in ("tier1", "tier2"):
                t = int(r["scope"][-1])
                lgs = [lg for lg, tt in tier_of.items() if tt == t and lg in per_league_improved]
                r["metrics"]["leagues_total"] = len(lgs)
                r["metrics"]["leagues_improved"] = int(sum(per_league_improved[lg] for lg in lgs))
    return rows


def passes(rows):
    """pooled improvement in both tiers AND >=4 of 5 leagues improved per tier."""
    ok = {}
    for r in rows:
        if r["scope"] in ("tier1", "tier2"):
            m = r["metrics"]
            ok[r["scope"]] = (m.get("d_vs_base", 1) < 0 and m.get("leagues_improved", 0) >= min(4, m.get("leagues_total", 5)))
    return len(ok) == 2 and all(ok.values())


def get_m(rows, scope):
    for r in rows:
        if r["scope"] == scope and r["league"] is None:
            return r["metrics"]
    return {}


# =============================================================== persistence
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


DDL = """
create table if not exists experiment_results (
  id bigserial primary key,
  batch text not null,
  run_name text not null,
  period text not null,
  scope text not null,
  league text,
  n integer,
  metrics jsonb not null,
  config jsonb,
  created_at timestamptz not null default now()
);
create index if not exists experiment_results_batch_idx on experiment_results (batch, run_name);
create table if not exists experiment_predictions (
  batch text not null,
  run_name text not null,
  match_id bigint not null,
  p_home real, p_draw real, p_away real, p_over25 real,
  primary key (batch, run_name, match_id)
);
alter table experiment_results enable row level security;
alter table experiment_predictions enable row level security;
"""


def save_db(url, batch, results, preds):
    import psycopg
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            cur.execute("delete from experiment_results where batch = %s", (batch,))
            cur.execute("delete from experiment_predictions where batch = %s", (batch,))
            cur.executemany(
                "insert into experiment_results (batch, run_name, period, scope, league, n, metrics, config) "
                "values (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)",
                [(batch, r["run"], r["period"], r["scope"], r["league"], r["metrics"].get("n"),
                  json.dumps(clean(r["metrics"])), json.dumps(clean(r["config"]))) for r in results])
            for name, p in preds.items():
                cur.executemany(
                    "insert into experiment_predictions values (%s,%s,%s,%s,%s,%s,%s) on conflict do nothing",
                    [(batch, name, int(a), float(b), float(c), float(d), None if not np.isfinite(e) else float(e))
                     for a, b, c, d, e in zip(p.mid, p.pH, p.pD, p.pA, p.pO)])
        conn.commit()


def fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_summary(path, results, batch, combo_spec, meta_info):
    lines = [f"# Experiments batch `{batch}`", "", meta_info, ""]
    if combo_spec is not None:
        lines += ["Chosen combo spec (rule: both tiers pooled improvement AND >=4/5 leagues per tier):", "",
                  "```", json.dumps(combo_spec), "```", ""]
    for period in sorted({r["period"] for r in results}):
        lines += [f"## Period: {period}", "",
                  "| run | t1 d_base [95% CI] | t1 lg | t2 d_base [95% CI] | t2 lg | t1 d_mkt [CI] | t2 d_mkt [CI] | ll t1 | ll t2 | mkt t1 | mkt t2 | CLV t1 | CLV t2 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        runs = []
        for r in results:
            if r["period"] == period and r["run"] not in runs:
                runs.append(r["run"])
        for run in runs:
            rr = [r for r in results if r["run"] == run and r["period"] == period]
            t1, t2 = get_m(rr, "tier1"), get_m(rr, "tier2")

            def cell(m, k="d_vs_base"):
                return f"{fmt(m.get(k), 4)} [{fmt(m.get(k + '_lo'), 4)}, {fmt(m.get(k + '_hi'), 4)}]"
            lines.append(
                f"| {run} | {cell(t1)} | {t1.get('leagues_improved', '-')}/{t1.get('leagues_total', '-')} | "
                f"{cell(t2)} | {t2.get('leagues_improved', '-')}/{t2.get('leagues_total', '-')} | "
                f"{cell(t1, 'd_vs_market')} | {cell(t2, 'd_vs_market')} | "
                f"{fmt(t1.get('ll'))} | {fmt(t2.get('ll'))} | {fmt(t1.get('ll_market'))} | {fmt(t2.get('ll_market'))} | "
                f"{fmt(t1.get('clv_mean'), 3)} (n={t1.get('clv_n', '-')}) | {fmt(t2.get('clv_mean'), 3)} (n={t2.get('clv_n', '-')}) |")
        lines.append("")
    lines += ["d_base < 0 = better than base (poisson_v1 replica). d_mkt > 0 = WORSE than the closing market.",
              "CLV = mean of (pre odds x closing fair prob - 1) over bets with model edge > threshold; > 0 means the model beats the closing line."]
    open(path, "w").write("\n".join(lines))


# =============================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["db", "synthetic"], default="db")
    ap.add_argument("--batch", default=time.strftime("dev_%Y%m%d_%H%M"))
    ap.add_argument("--pred-from", default="2022-08-01")
    ap.add_argument("--eval-from", default="2023-08-01")
    ap.add_argument("--dev-to", default="2025-06-30")
    ap.add_argument("--holdout", action="store_true", help="also evaluate 2025-07-01 onward (run ONCE)")
    ap.add_argument("--train-years", type=float, default=4.0)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--runs", default="all", help="comma list, or 'all'")
    ap.add_argument("--combo-spec", default=None, help="JSON, fixes the combo instead of auto-selecting")
    ap.add_argument("--store-preds", default="", help="comma list of runs whose predictions go to the DB")
    ap.add_argument("--no-db-write", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    t0 = time.time()
    if args.source == "db":
        url = os.environ.get("DATABASE_URL")
        if not url:
            sys.exit("set DATABASE_URL first (export DATABASE_URL='...') or use --source synthetic")
        df = load_db(url)
    else:
        url = None
        df = make_synthetic()
        args.no_db_write = True
    df = add_rest(add_elo(df))
    df["tot"] = df.hg + df.ag
    print(f"loaded {len(df)} matches, {df.league.nunique()} leagues, "
          f"{df.match_date.min().date()} .. {df.match_date.max().date()}; "
          f"closing odds on {df.mc_H.notna().mean():.0%}, pinnacle {df.pc_H.notna().mean():.0%}, "
          f"shots {df.hs.notna().mean():.0%}")

    dev_to = pd.Timestamp(args.dev_to)
    pred_to = df.match_date.max() if args.holdout else dev_to
    eval_from = pd.Timestamp(args.eval_from)
    periods = {"dev": (eval_from, dev_to)}
    if args.holdout:
        periods = {"holdout": (dev_to + pd.Timedelta(days=1), df.match_date.max())}
        # on a hold-out run only the hold-out period is reported
    meta = df[["mid", "league", "country", "tier", "match_date", "result", "tot", "elo_diff", "rest_diff",
               "mc_H", "mc_D", "mc_A", "pc_H", "pc_D", "pc_A", "mp_H", "mp_D", "mp_A",
               "ao_H", "ao_D", "ao_A", "ouc_O", "oup_O"]].copy()

    cache = {}

    def base_pred(cfg):
        k = json.dumps(cfg, sort_keys=True)
        if k not in cache:
            ts = time.time()
            cache[k] = run_base_model(df, cfg, pd.Timestamp(args.pred_from), pred_to,
                                      args.step_days, args.train_years, args.jobs)
            print(f"  fitted {k} -> {len(cache[k])} preds in {time.time() - ts:.0f}s", flush=True)
        return cache[k]

    def cfg_with(**kw):
        c = dict(BASE_CFG)
        c.update(kw)
        return c

    experiments = [  # name, model cfg, stack feats (None = no stacking)
        ("base", cfg_with(), None),
        ("hl90", cfg_with(hl=90), None),
        ("hl365", cfg_with(hl=365), None),
        ("ridge_lo", cfg_with(ridge=0.05), None),
        ("ridge_hi", cfg_with(ridge=1.5), None),
        ("dc", cfg_with(dc=True), None),
        ("joint", cfg_with(joint=True), None),
        ("xg30", cfg_with(xg_w=0.3), None),
        ("xg60", cfg_with(xg_w=0.6), None),
        # ---- round 2 (select with --runs): stronger xG weight and combinations
        ("xg80", cfg_with(xg_w=0.8), None),
        ("xg100", cfg_with(xg_w=1.0), None),
        ("xg60_hl365", cfg_with(xg_w=0.6, hl=365), None),
        ("xg60_hl365_r15", cfg_with(xg_w=0.6, hl=365, ridge=1.5), None),
        ("xg80_hl365_r15", cfg_with(xg_w=0.8, hl=365, ridge=1.5), None),
        ("xg100_hl365_r15", cfg_with(xg_w=1.0, hl=365, ridge=1.5), None),
        ("xg80_hl365_r15_calib", cfg_with(xg_w=0.8, hl=365, ridge=1.5), []),
        ("xg80_hl365_r15_elo_rest", cfg_with(xg_w=0.8, hl=365, ridge=1.5), ["elo", "rest"]),
        ("xg80_hl365_r15_mkt", cfg_with(xg_w=0.8, hl=365, ridge=1.5), ["mkt"]),
        ("elo", cfg_with(), ["elo"]),
        ("rest", cfg_with(), ["rest"]),
        ("calib", cfg_with(), []),
        ("mkt", cfg_with(), ["mkt"]),
    ]
    want = None if args.runs == "all" else set(args.runs.split(","))

    results, preds_keep, run_rows = [], {}, {}
    keep_preds = set(x for x in args.store_preds.split(",") if x)

    def do_run(name, cfg, feats):
        print(f"[{name}] cfg={cfg} feats={feats}", flush=True)
        p = base_pred(cfg)
        if feats is not None:
            p = stack(p, meta, feats)
        preds_keep[name] = p
        for period, rng in periods.items():
            rows = evaluate(name, p, meta, rng, base_ll_map.get(period, {}), args.bootstrap, args.edge, name == "base")
            run_rows[(name, period)] = rows
            for r in rows:
                results.append({"run": name, "period": period, "scope": r["scope"], "league": r["league"],
                                "metrics": r["metrics"], "config": {"cfg": cfg, "feats": feats}})
            t1, t2 = get_m(rows, "tier1"), get_m(rows, "tier2")
            print(f"    {period}: t1 ll={fmt(t1.get('ll'))} (mkt {fmt(t1.get('ll_market'))}) "
                  f"d_base={fmt(t1.get('d_vs_base'))} lg={t1.get('leagues_improved', '-')}/{t1.get('leagues_total', '-')} | "
                  f"t2 ll={fmt(t2.get('ll'))} (mkt {fmt(t2.get('ll_market'))}) "
                  f"d_base={fmt(t2.get('d_vs_base'))} lg={t2.get('leagues_improved', '-')}/{t2.get('leagues_total', '-')}",
                  flush=True)

    # base first (needed as reference)
    base_ll_map = {}
    bp = base_pred(BASE_CFG)
    for period, rng in periods.items():
        ev = eval_frame(bp, meta)
        ev = ev[(ev.match_date >= rng[0]) & (ev.match_date <= rng[1])]
        base_ll_map[period] = ev.set_index("mid")["ll"].to_dict()

    for name, cfg, feats in experiments:
        if want is None or name in want or name == "base":
            do_run(name, cfg, feats)

    # ---------------------------------------------------------------- combo
    combo_spec = None
    if want is None or "combo" in want or "combo_mkt" in want:
        if args.combo_spec:
            combo_spec = json.loads(args.combo_spec)
        else:
            def ok(n):
                return (n, "dev") in run_rows and passes(run_rows[(n, "dev")])

            def gain(n):
                return get_m(run_rows[(n, "dev")], "all").get("d_vs_base", 0)
            combo_spec = dict(BASE_CFG)
            hl_c = [n for n in ("hl90", "hl365") if ok(n)]
            if hl_c:
                combo_spec["hl"] = {"hl90": 90, "hl365": 365}[min(hl_c, key=gain)]
            rg_c = [n for n in ("ridge_lo", "ridge_hi") if ok(n)]
            if rg_c:
                combo_spec["ridge"] = {"ridge_lo": 0.05, "ridge_hi": 1.5}[min(rg_c, key=gain)]
            combo_spec["dc"] = ok("dc")
            combo_spec["joint"] = ok("joint")
            xg_c = [n for n in ("xg30", "xg60") if ok(n)]
            if xg_c:
                combo_spec["xg_w"] = {"xg30": 0.3, "xg60": 0.6}[min(xg_c, key=gain)]
            combo_spec["feats"] = [n for n in ("elo", "rest") if ok(n)]
        print(f"combo spec: {combo_spec}")
        cfg = {k: combo_spec[k] for k in BASE_CFG}
        feats = list(combo_spec.get("feats", []))
        if want is None or "combo" in want:
            do_run("combo", cfg, feats)
        if want is None or "combo_mkt" in want:
            do_run("combo_mkt", cfg, feats + ["mkt"])

    # ---------------------------------------------------------------- output
    os.makedirs(os.path.join(args.out, args.batch), exist_ok=True)
    outdir = os.path.join(args.out, args.batch)
    flat = [{"run": r["run"], "period": r["period"], "scope": r["scope"], "league": r["league"], **clean(r["metrics"])}
            for r in results]
    pd.DataFrame(flat).to_csv(os.path.join(outdir, "results.csv"), index=False)
    info = (f"source={args.source}, pred_from={args.pred_from}, eval={args.eval_from}..{args.dev_to}"
            f"{' + holdout' if args.holdout else ''}, step={args.step_days}d, train_years={args.train_years}, "
            f"bootstrap={args.bootstrap}, matches={len(df)}")
    write_summary(os.path.join(outdir, "summary.md"), results, args.batch, combo_spec, info)
    print(f"\nwrote {outdir}/summary.md and results.csv")
    if url and not args.no_db_write:
        keep = {n: preds_keep[n] for n in keep_preds if n in preds_keep}
        save_db(url, args.batch, results, keep)
        print(f"saved to Supabase: experiment_results (batch={args.batch})")
    print(f"done in {time.time() - t0:.0f}s")
    print(open(os.path.join(outdir, "summary.md")).read())


if __name__ == "__main__":
    main()
