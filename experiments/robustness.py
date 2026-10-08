"""
일반화 검증 (v2.2): "내가 준 기간 말고 다른 과거 데이터를 넣어도 비슷하게 나오나?"
  1) 롤링 검증: 2개월마다 기준 달을 옮기며 '그 전 데이터 전부로 학습 → 다음 2개월 예측' (8회)
  2) 거꾸로 검증: 2026년(1~9월)으로 학습 → 2025년(3~12월) 예측 (시간 방향을 뒤집어도 되는지)
  3) 처음 보는 항로: 항로 20%를 학습에서 통째로 빼고, 그 항로 배만 예측
지표: 30분 안 비율, 1시간 안 비율, 평균 오차(분). 터미널 예정 / 같은 입력 단순 평균 / LSTM
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd
from data.features import load_samples, ArrivalScaler
from data.storage import latest_upload
from serving_app.train_and_register import fit_base, predict

S = load_samples(latest_upload()); sc = ArrivalScaler.load("serving_app/models/scaler.pkl")
def stats(y, p):
    e = np.abs(np.asarray(y) - np.asarray(p))
    return {"n": len(e), "within_30m": round(float((e <= .5).mean()), 3), "within_1h": round(float((e <= 1).mean()), 3),
            "mae_min": round(float(e.mean() * 60), 1)}
def evaluate(train, test, name):
    y = [s["actual_correction"] for s in test]
    m = fit_base(train, sc)
    r = {"case": name, "n_train": len(train),
         "terminal": stats(y, [0] * len(y)),
         "mean20": stats(y, [np.mean([q["delay_hours"] for q in s["sequence"]]) for s in test]),
         "lstm": stats(y, predict(m, test, sc))}
    print(json.dumps(r, ensure_ascii=False), flush=True)
    return r
out = []
T = lambda s: s["decision_time"]
for start in pd.date_range("2025-05-01", "2026-09-01", freq="2MS"):
    end = start + pd.DateOffset(months=2)
    tr = [s for s in S if T(s) < start]; te = [s for s in S if start <= T(s) < end]
    if len(tr) >= 200 and len(te) >= 30:
        out.append(evaluate(tr, te, f"rolling {start:%Y-%m}~{(end - pd.Timedelta(days=1)):%Y-%m}"))
tr = [s for s in S if T(s) >= pd.Timestamp("2026-01-01")]; te = [s for s in S if pd.Timestamp("2025-03-01") <= T(s) < pd.Timestamp("2026-01-01")]
out.append(evaluate(tr, te, "reverse: train 2026 -> test 2025"))
rng = np.random.default_rng(7); routes = sorted({s["route"] for s in S}); held = set(rng.choice(routes, size=len(routes) // 5, replace=False))
tr = [s for s in S if s["route"] not in held and T(s) < pd.Timestamp("2026-05-01")]
te = [s for s in S if s["route"] in held and T(s) >= pd.Timestamp("2026-05-01")]
out.append(evaluate(tr, te, f"unseen routes ({len(held)} of {len(routes)})"))
os.makedirs("reports/robustness_v2.2", exist_ok=True)
json.dump(out, open("reports/robustness_v2.2/results.json", "w"), ensure_ascii=False, indent=1)
