"""작업시간 추정 개선 후보 비교 (테스트 2026-05~). schedule.py 개선 근거."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from data.hjnc import read_berth_excel, completed_calls
from data.storage import latest_upload
from data.features import VAL_END
d = completed_calls(read_berth_excel(latest_upload()))
tr = d[d["입항일시"] < VAL_END]
A = np.stack([np.ones(len(tr)), tr["box"].values], 1); (a, b), *_ = np.linalg.lstsq(A, tr["work_h"].values, rcond=None)
d["base"] = np.clip(a + b * d["box"], 1, None); d["lr"] = np.log(d["work_h"] / d["base"])
te = d[d["입항일시"] >= VAL_END]
res = {k: [] for k in ["base", "route10", "route5", "route10+berth", "route10+recent14d", "route_median10", "shrink"]}
for _, r in te.iterrows():
    t = r["입항일시"] - pd.Timedelta(hours=10)
    past = d[d["작업 완료일시"] < t]
    g = past[past["항로"] == r["항로"]]["lr"]
    bth = past[past["선석"] == r["선석"]]["lr"].tail(30)
    rec = past[past["작업 완료일시"] >= t - pd.Timedelta(days=14)]["lr"]
    m10 = g.tail(10).mean() if len(g) else 0.0
    res["base"].append(r["base"])
    res["route10"].append(r["base"] * np.exp(m10))
    res["route5"].append(r["base"] * np.exp(g.tail(5).mean() if len(g) else 0.0))
    res["route_median10"].append(r["base"] * np.exp(g.tail(10).median() if len(g) else 0.0))
    res["route10+berth"].append(r["base"] * np.exp(m10 + 0.5 * (bth.mean() if len(bth) else 0) - 0.5 * (past["lr"].tail(200).mean())))
    res["route10+recent14d"].append(r["base"] * np.exp(m10 + (rec.mean() - past["lr"].tail(200).mean() if len(rec) else 0)))
    n = len(g.tail(10)); res["shrink"].append(r["base"] * np.exp(m10 * n / (n + 3) if n else 0.0))
y = te["work_h"].values
for k, p in res.items():
    e = np.abs(y - np.array(p)); print(f"{k:20s} MAE {e.mean():.3f}h  median {np.median(e):.2f}h  >4h {(e>4).mean():.2%}")
