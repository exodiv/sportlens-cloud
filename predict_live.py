#!/usr/bin/env python3
"""
SportLens - live predictions with the locked model v2  ("poisson_v2" v1.0)

Reads upcoming fixtures from Supabase (filled daily by the in-database job
ingest_fixtures()), predicts them with the model chosen on the DEV period and confirmed
on the 2025/26 HOLD-OUT, and stores the result in the immutable `predictions` table
(the database trigger refuses inserts for matches that are today or already played, and
refuses any update/delete - so every stored prediction is a true before-the-match record).

Model spec (frozen):  half-life 365 d, ridge 1.5, pseudo-xG weight 0.8, Dixon-Coles off,
per-league fit, then logistic stacking on [log-odds of the Poisson probs, Elo diff, rest diff]
trained on earlier out-of-sample predictions (per tier). Over/under 2.5 is calibrated the same way.

Usage
-----
  export DATABASE_URL='postgresql://...'     # in your own terminal
  python3 predict_live.py --dry-run          # show predictions, write nothing
  python3 predict_live.py                    # write predictions for matches in the next 2 days
  python3 predict_live.py --horizon-days 3
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression

import experiments_v2 as ex

SPEC = {"hl": 365, "dc": False, "joint": False, "xg_w": 0.8, "ridge": 1.5, "feats": ["elo", "rest"]}
MODEL_NAME, MODEL_VERSION = "poisson_v2", "1.0"
STEP = 21          # days per refit window
N_WINDOWS = 45     # ~2.6 years of out-of-sample history for the stacking layer
TRAIN_DAYS = 4 * 365


def base_predictions(df, today, jobs):
    cfg = {k: SPEC[k] for k in ex.BASE_CFG}
    dates = list(pd.date_range(end=today, periods=N_WINDOWS, freq=f"{STEP}D"))
    tasks = [(g.sort_values("match_date"), cfg, dates, STEP, TRAIN_DAYS) for _, g in df.groupby("league")]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            parts = list(pool.map(ex.run_group, tasks))
    else:
        parts = [ex.run_group(t) for t in tasks]
    return pd.concat([p for p in parts if len(p)], ignore_index=True)


def live_stack(pred, meta, fixture_ids, feats, min_n=600, C=10.0):
    cols = ["mid", "tier", "result", "tot", "elo_diff", "rest_diff", "mp_H", "mp_D", "mp_A", "oup_O"]
    d = pred.merge(meta[cols], on="mid")
    is_fx = d.mid.isin(fixture_ids)
    known, new = d[~is_fx & d.result.notna()], d[is_fx]
    out = []
    for tier in (1, 2):
        tr, te = known[known.tier == tier], new[new.tier == tier]
        if te.empty:
            continue
        Xtr, Xte = ex.feat_matrix(tr, feats), ex.feat_matrix(te, feats)
        ok = np.isfinite(Xtr).all(1)
        if ok.sum() < min_n:
            print(f"  tier {tier}: only {ok.sum()} training rows -> using unstacked probabilities")
            res = te[["mid", "pH", "pD", "pA", "pO"]].copy()
            out.append(res)
            continue
        clf = LogisticRegression(C=C, max_iter=500).fit(Xtr[ok], tr.result.to_numpy()[ok])
        ci = {c: i for i, c in enumerate(clf.classes_)}
        pr = clf.predict_proba(np.nan_to_num(Xte))
        res = pd.DataFrame({"mid": te.mid.to_numpy(), "pH": pr[:, ci["H"]], "pD": pr[:, ci["D"]],
                            "pA": pr[:, ci["A"]], "pO": te.pO.to_numpy()})
        yo = (tr.tot.to_numpy()[ok] > 2.5).astype(int)
        co = LogisticRegression(C=C, max_iter=500).fit(ex.logit(tr.pO.to_numpy()[ok]).reshape(-1, 1), yo)
        res["pO"] = co.predict_proba(ex.logit(te.pO.to_numpy()).reshape(-1, 1))[:, 1]
        out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["mid", "pH", "pD", "pA", "pO"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon-days", type=int, default=2, help="predict matches dated tomorrow .. today+N")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("set DATABASE_URL first: export DATABASE_URL='...'")
    today = pd.Timestamp(datetime.now(timezone.utc).date())

    df = ex.load_db(url, live=True)
    df["tot"] = df.hg + df.ag
    df = ex.add_rest(ex.add_elo(df))
    played = df[df.hg.notna()]
    fx = df[df.hg.isna() & (df.match_date > today) & (df.match_date <= today + pd.Timedelta(days=args.horizon_days))]
    print(f"{len(played)} played matches, last result {played.match_date.max().date()}; "
          f"{len(fx)} fixtures in the next {args.horizon_days} day(s)")
    if fx.empty:
        print("nothing to predict (fixtures.csv has no matches in the window yet) - try again tomorrow")
        return

    import psycopg
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute("select id from model_versions where name=%s and version=%s", (MODEL_NAME, MODEL_VERSION))
            r = cur.fetchone()
            mv = r[0] if r else None
            done = set()
            if mv:
                cur.execute("select match_id from predictions where model_version_id=%s and market='1X2'", (mv,))
                done = {x[0] for x in cur.fetchall()}
    fx = fx[~fx.mid.isin(done)]
    if fx.empty:
        print("all fixtures in the window already have a stored prediction")
        return

    # pre-match elo/rest of the fixtures were computed inside add_elo/add_rest (no goals needed)
    allp = base_predictions(df, today, args.jobs)
    meta = df[["mid", "tier", "result", "tot", "elo_diff", "rest_diff", "mp_H", "mp_D", "mp_A", "oup_O"]]
    res = live_stack(allp, meta, set(fx.mid), SPEC["feats"])
    res = res.merge(df[["mid", "league", "match_date", "home_id", "away_id", "mp_H", "mp_D", "mp_A"]], on="mid")
    names = {}
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("select id, name from teams where id = any(%s)", (list(set(res.home_id) | set(res.away_id)),))
        names = dict(cur.fetchall())
    res["match"] = res.home_id.map(names) + " - " + res.away_id.map(names)
    show = res.sort_values(["match_date", "league"])[["league", "match_date", "match", "pH", "pD", "pA", "pO", "mp_H", "mp_D", "mp_A"]]
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(show.round(3).to_string(index=False))
    print(f"\n{len(res)} predictions ready (mp_* = market probability before the match, de-margined)")

    if args.dry_run:
        print("dry run: nothing written")
        return
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            if not mv:
                cur.execute("insert into model_versions (name, version, notes) values (%s,%s,%s) returning id",
                            (MODEL_NAME, MODEL_VERSION, "frozen spec: " + json.dumps(SPEC)))
                mv = cur.fetchone()[0]
            n = 0
            for r in res.itertuples():
                cur.execute("insert into predictions (match_id, model_version_id, market, probabilities) "
                            "values (%s,%s,'1X2',%s) on conflict do nothing",
                            (int(r.mid), mv, json.dumps({"H": round(r.pH, 5), "D": round(r.pD, 5), "A": round(r.pA, 5)})))
                n += cur.rowcount
                if np.isfinite(r.pO):
                    cur.execute("insert into predictions (match_id, model_version_id, market, probabilities) "
                                "values (%s,%s,'OU2.5',%s) on conflict do nothing",
                                (int(r.mid), mv, json.dumps({"O": round(r.pO, 5), "U": round(1 - r.pO, 5)})))
        conn.commit()
    print(f"stored {n} new 1X2 predictions (model {MODEL_NAME} v{MODEL_VERSION})")


if __name__ == "__main__":
    main()
