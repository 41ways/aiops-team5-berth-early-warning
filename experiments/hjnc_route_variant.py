"""hjnc_residual_test.py 변형: 입력 20척을 '같은 항로(또는 선사)' 이력으로. 부족하면 터미널 전체로 채움."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd
import experiments.hjnc_residual_test as T
from data.hjnc import completed_calls, read_berth_excel

def hist_for(prev, r, key):
    g = prev[prev[key] == r[key]]
    if len(g) >= T.SEQ:
        return g.tail(T.SEQ)
    return pd.concat([prev[prev[key] != r[key]].tail(T.SEQ - len(g)), g])

def arrival(d, key):
    d = d.sort_values("입항일시").reset_index(drop=True)
    d["dev"] = (d["입항일시"] - d["plan_berth_10h"]).dt.total_seconds() / 3600
    rows = []
    for i in range(T.SEQ, len(d)):
        r = d.iloc[i]; prev = d.iloc[:i]
        prev = prev[prev["입항일시"] <= r["plan_berth_10h"] - pd.Timedelta(hours=10)]
        if len(prev) < T.SEQ: continue
        h = hist_for(prev, r, key)
        seq = np.stack([(h.dev.values + 6) / 12, h.box.values / 8000], 1)
        rows.append({"as_of": r["plan_berth_10h"], "seq": seq, "target": r.dev,
                     "grp_med": prev.loc[prev[key] == r[key], "dev"].tail(10).median() if (prev[key] == r[key]).any() else prev.dev.median()})
    s = pd.DataFrame(rows)
    te, p, ntr, nva = T.fit_eval(s, (-6, 6), "arrival")
    med = s[s.as_of < T.TR_END].target.median()
    print("A", key, {"plan": round(np.mean(np.abs(te.target)),3), "global_bias": round(np.mean(np.abs(te.target-med)),3),
                     "group_last10_median": round(np.mean(np.abs(te.target-te.grp_med)),3),
                     "LSTM": round(np.mean(np.abs(te.target-p)),3), "LSTM+bias_avg": round(np.mean(np.abs(te.target-(p+med)/2)),3)}, flush=True)

def worktime(d, key):
    tr0 = d[d["입항일시"] < T.TR_END]
    w, *_ = np.linalg.lstsq(np.stack([np.ones(len(tr0)), tr0.box.values], 1), tr0.work_h.values, rcond=None)
    d = d.copy(); d["base"] = np.clip(w[0] + w[1] * d.box.values, 1.0, None); d["lr"] = np.log(d.work_h / d.base)
    rows = []
    for i, r in d.iterrows():
        prev = d[d["작업 완료일시"] < r["입항일시"]]
        if len(prev) < T.SEQ: continue
        h = hist_for(prev, r, key)
        seq = np.stack([(h.lr.values + 1.5) / 3, h.box.values / 8000], 1)
        g = prev[prev[key] == r[key]]
        rows.append({"as_of": r["입항일시"], "seq": seq, "target": r.lr, "base": r.base, "work_h": r.work_h,
                     "grp": g.lr.tail(10).mean() if len(g) else 0.0})
    s = pd.DataFrame(rows)
    te, p, ntr, nva = T.fit_eval(s, (-1.5, 1.5), "worktime")
    print("B", key, {"linear": round(np.mean(np.abs(te.base-te.work_h)),3),
                     "linear*group_last10": round(np.mean(np.abs(te.base*np.exp(te.grp)-te.work_h)),3),
                     "linear*LSTM": round(np.mean(np.abs(te.base*np.exp(p)-te.work_h)),3)}, flush=True)

d = completed_calls(read_berth_excel(sys.argv[1]))
for key in ("항로", "선사"):
    arrival(d, key); worktime(d, key)
