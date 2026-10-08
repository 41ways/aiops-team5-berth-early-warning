"""
일정 체인 신호 확인: '같은 선석 앞 배가 늦게 끝날 것 같으면 이번 배 접안도 늦는가?'
예측 시점(반입 마감) 기준으로 같은 선석의 바로 앞 배 상태를 본다 (그 시점에 알 수 있는 것만):
  - 이미 출항했으면 여유 큼
  - 작업 중이면 예상 작업완료 = 실제 작업시작 + 예측 작업시간 (실제 작업시간은 안 씀)
  - 아직 안 붙었으면 예상 작업완료 = 계획 접안 + 접안→작업시작 + 예측 작업시간
  gap_h = 이번 배 계획 접안 − 앞 배 예상 작업완료 (작을수록 빠듯)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd
from tensorflow import keras
from data.hjnc import read_berth_excel
from data.features import load_samples, split_by_period, ArrivalScaler, VAL_END
from data.storage import latest_upload
from serving_app.schedule import WorkTimeEstimator
from serving_app.train_and_register import predict

path = latest_upload(); df = read_berth_excel(path)
est = WorkTimeEstimator(df, fit_before=VAL_END)
ids = df.drop_duplicates("모선항차", keep="last").set_index("모선항차")
S = load_samples(path)
sc = ArrivalScaler.load("serving_app/models/scaler.pkl"); m = keras.models.load_model("serving_app/models/arrival_v1.keras")
pred = np.array(predict(m, S, sc))
rows = []
d_all = df[df["box"] > 0].copy()
for s, p in zip(S, pred):
    t = s["decision_time"]; r = ids.loc[s["call_id"]]
    same = d_all[(d_all["선석"] == r["선석"]) & (d_all["plan_berth_10h"] < s["planned_berth"]) & (d_all["모선항차"] != s["call_id"])]
    if same.empty: continue
    pv = same.sort_values("plan_berth_10h").iloc[-1]
    if pd.notna(pv["출항일시"]) and pd.notna(pv["작업 완료일시"]) and pv["작업 완료일시"] <= t:
        fin = pv["작업 완료일시"]; state = "done"
    elif pd.notna(pv["작업 시작일시"]) and pv["작업 시작일시"] <= t:
        fin = pv["작업 시작일시"] + pd.Timedelta(hours=est.work_hours(pv["box"], pv["항로"], t)); state = "working"
    else:
        fin = pv["plan_berth_10h"] + pd.Timedelta(hours=est.pre_h + est.work_hours(pv["box"], pv["항로"], t)); state = "not_berthed"
    fin = max(fin, t)
    rows.append({"dt": t, "y": s["actual_correction"], "p": p, "gap": (s["planned_berth"] - fin).total_seconds() / 3600, "state": state})
v = pd.DataFrame(rows); v["res"] = v["y"] - v["p"]
v["gapb"] = pd.cut(v["gap"], [-1e9, 0, 1, 2, 4, 8, 1e9], labels=["<0", "0-1", "1-2", "2-4", "4-8", "8+"])
print("all n", len(v), "corr(res, gap)", v[["res", "gap"]].corr().iloc[0, 1].round(3))
print(v.groupby("gapb", observed=True).agg(n=("res", "size"), y_mean=("y", "mean"), res_mean=("res", "mean"), res_mae=("res", lambda x: x.abs().mean())).round(3))
print(v.groupby("state").agg(n=("res", "size"), res_mean=("res", "mean")).round(3))
te = v[v["dt"] >= VAL_END]
print("test by gap:"); print(te.groupby("gapb", observed=True).agg(n=("res", "size"), res_mean=("res", "mean"), res_mae=("res", lambda x: x.abs().mean())).round(3))
v.to_csv("/tmp/claude-0/-home-claude/28eae318-ce0d-52d9-8d27-3a139074bf72/scratchpad/chain.csv", index=False)
