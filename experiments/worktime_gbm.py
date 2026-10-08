"""
작업시간(→ 출항) 예측 개선 실험 v1.4.
기준: 현재 schedule.py (물량 선형식 × 항로 최근 10척 보정)
후보: 같은 정보 + 동시 작업 배 수(크레인 나눠 쓰기) 등을 넣은 HistGradientBoosting (LightGBM과 같은 계열 트리 부스팅)
분할: 학습 2025 / 검증 2026-01~04 (선택) / 테스트 2026-05~ (보고)
모든 특성은 예측 시점(반입 마감 = 계획 접안 10시간 전)에 알 수 있는 값만. 다른 배의 '실제 작업시간'은 쓰지 않는다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from data.hjnc import read_berth_excel, completed_calls
from data.storage import latest_upload
from data.features import TRAIN_END, VAL_END

df = read_berth_excel(latest_upload())
d = completed_calls(df).sort_values("입항일시").reset_index(drop=True)
fit = d[d["입항일시"] < TRAIN_END]
A = np.stack([np.ones(len(fit)), fit["box"].values], 1); (a, b), *_ = np.linalg.lstsq(A, fit["work_h"].values, rcond=None)
d["base"] = np.clip(a + b * d["box"], 1, None)
d["lr"] = np.log(d["work_h"] / d["base"])
d["asof"] = d["plan_berth_10h"] - pd.Timedelta(hours=10)
# 다른 배의 '계획상' 작업 구간: 계획 접안 ~ 계획 접안 + 기본 작업시간 (실제 작업시간은 안 씀)
all_ = df[df["box"] > 0].copy()
all_["base"] = np.clip(a + b * all_["box"], 1, None)
all_["ps"] = all_["plan_berth_10h"]; all_["pe"] = all_["ps"] + pd.to_timedelta(all_["base"], unit="h")
ps, pe, pbox, pid = all_["ps"].values, all_["pe"].values, all_["box"].values, all_["모선항차"].values

rows = []
for i, r in d.iterrows():
    t = r["asof"]
    past = d[d["작업 완료일시"] < t]
    g = past[past["항로"] == r["항로"]]["lr"]
    rec = past[past["작업 완료일시"] >= t - pd.Timedelta(days=14)]["lr"]
    s, e = r["plan_berth_10h"], r["plan_berth_10h"] + pd.Timedelta(hours=r["base"])
    ov = (ps < np.datetime64(e)) & (pe > np.datetime64(s)) & (pid != r["모선항차"])
    rows.append({"route10": g.tail(10).mean() if len(g) else 0.0, "route_n": min(len(g), 10),
                 "route_last": g.iloc[-1] if len(g) else 0.0, "recent14": rec.mean() if len(rec) else 0.0,
                 "conc_n": int(ov.sum()), "conc_box": float(pbox[ov].sum()),
                 "box": r["box"], "dis_share": r["양하"] / max(r["box"], 1), "sh_share": r["S/H"] / max(r["box"], 1),
                 "berth": {"1B": 1, "2B": 2, "3B": 3, "4B": 4}.get(r["선석"], 0),
                 "hour": r["plan_berth_10h"].hour, "dow": r["plan_berth_10h"].dayofweek})
X = pd.DataFrame(rows)
y = d["lr"].values
tr = (d["입항일시"] < TRAIN_END).values; va = ((d["입항일시"] >= TRAIN_END) & (d["입항일시"] < VAL_END)).values
te = (d["입항일시"] >= VAL_END).values
work = d["work_h"].values; base = d["base"].values

def mae(pred_lr, m): return np.abs(work[m] - base[m] * np.exp(pred_lr[m])).mean()
print(f"n train {tr.sum()} val {va.sum()} test {te.sum()}")
print(f"base only          val {mae(np.zeros(len(d)), va):.3f}  test {mae(np.zeros(len(d)), te):.3f}")
print(f"route10 (현재)     val {mae(X['route10'].values, va):.3f}  test {mae(X['route10'].values, te):.3f}")
feats_sets = {"route+conc": ["route10", "route_n", "conc_n", "conc_box"],
              "all": list(X.columns)}
for name, fs in feats_sets.items():
    for loss in ("squared_error", "absolute_error"):
        res = []
        for seed in (1, 2, 3):
            m = HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
                                              categorical_features=[fs.index("berth")] if "berth" in fs else None)
            m.fit(X.loc[tr, fs], y[tr]); p = m.predict(X[fs]); res.append((mae(p, va), mae(p, te)))
        r = np.array(res).mean(0); print(f"GBM {name:10s} {loss:15s} val {r[0]:.3f}  test {r[1]:.3f}")
print("corr(lr, conc_n) train", np.corrcoef(y[tr], X.loc[tr, 'conc_n'])[0, 1].round(3))
