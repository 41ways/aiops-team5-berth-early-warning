"""
백테스트: "특정 기간까지 학습 → 그 다음 날들을 하루씩 예측" (2026-05-01 ~ 마지막 완료 선박).

각 배는 반입 마감 시점(접안 약 10시간 전)에 예측한다. 그 시점까지 접안한 배의 이력만 쓰므로 미래 정보가 섞이지 않는다.

비교 대상
  terminal : 터미널 예정 그대로 (반입 마감 + 10h)
  v1       : 2025년만으로 학습한 Production 모델 (serving_app/models/arrival_v1.keras)
  v1.1     : 2025-01 ~ 2026-04 (학습+검증 구간)으로 다시 학습한 후보. 시드 1·2·3 평균
  mean20   : 모델과 같은 입력(같은 항로 최근 20척 접안 오차)의 단순 평균만큼 보정 — LSTM이 이것보다 나은지 확인용
출항(선석 반납) 예측
  (출항일시는 완료 선박도 정각 단위 → 터미널이 확정한 출항 시각. 실측에 가까운 정답은 작업완료일시)
  naive    : 예측 접안 + 2025년 체류시간 중앙값
  schedule : 예측 접안 + 접안→작업시작 + 물량·항로 기반 작업시간 + 작업완료→출항 (serving_app/schedule.py)

실행: python scripts/backtest.py [엑셀]   → reports/backtest_v1.0/ 에 daily.csv, monthly.csv, vessels.csv, summary.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from tensorflow import keras

from data.features import VAL_END, ArrivalScaler, load_samples
from data.hjnc import completed_calls, read_berth_excel
from data.storage import latest_upload
from serving_app import train_and_register as tr
from serving_app.schedule import WorkTimeEstimator

OUT = "reports/backtest_v1.0"


def err_stats(e: np.ndarray) -> dict:
    e = np.abs(e)
    return {"mae_h": round(float(e.mean()), 3), "within_30m": round(float((e <= 0.5).mean()), 3),
            "over_1h": round(float((e > 1).mean()), 4), "max_h": round(float(e.max()), 2)}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_upload()
    os.makedirs(OUT, exist_ok=True)
    scaler = ArrivalScaler.load(tr.SCALER_PATH)
    samples = load_samples(path)
    trainval = [s for s in samples if s["decision_time"] < VAL_END]
    test = [s for s in samples if s["decision_time"] >= VAL_END]

    v1 = keras.models.load_model("serving_app/models/arrival_v1.keras")
    p_v1 = np.array(tr.predict(v1, test, scaler))
    p_v11 = []
    for seed in (1, 2, 3):
        tr.SEED = seed
        p_v11.append(tr.predict(tr.fit_base(trainval, scaler), test, scaler))
    tr.SEED = 1
    p_v11 = np.array(p_v11)

    df = read_berth_excel(path)
    est = WorkTimeEstimator(df, fit_before=VAL_END)
    done = completed_calls(df).set_index("모선항차")
    med_dwell = float(done[done["입항일시"] < VAL_END]["berth_h"].median())

    rows = []
    for i, s in enumerate(test):
        r = done.loc[s["call_id"]] if s["call_id"] in done.index else None
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        pred_berth = s["planned_berth"] + pd.Timedelta(hours=float(p_v1[i]))
        row = {"call_id": s["call_id"], "vessel": s["vessel"], "route": s["route"], "berth": s["berth"],
               "day": s["planned_berth"].normalize().date(), "month": s["planned_berth"].strftime("%Y-%m"),
               "planned_berth": s["planned_berth"], "actual_berth": s["actual_berth"],
               "err_terminal": s["actual_correction"], "err_v1": s["actual_correction"] - p_v1[i],
               "err_v11": s["actual_correction"] - p_v11[:, i].mean(),
               "err_mean20": s["actual_correction"] - float(np.mean([q["delay_hours"] for q in s["sequence"]]))}
        if r is not None:
            dep = est.departure(pred_berth, r["box"], s["route"], s["decision_time"])
            row.update(box=int(r["box"]), actual_departure=r["출항일시"], predicted_departure=dep,
                       err_dep_schedule=(r["출항일시"] - dep).total_seconds() / 3600,
                       err_dep_naive=(r["출항일시"] - (pred_berth + pd.Timedelta(hours=med_dwell))).total_seconds() / 3600,
                       err_work=r["work_h"] - est.work_hours(r["box"], s["route"], s["decision_time"]),
                       err_complete=(r["작업 완료일시"] - (dep - pd.Timedelta(hours=est.post_h))).total_seconds() / 3600)
        rows.append(row)
    v = pd.DataFrame(rows)
    v.to_csv(f"{OUT}/vessels.csv", index=False, encoding="utf-8-sig")

    def agg(g):
        return pd.Series({"vessels": len(g), "terminal_mae_h": g["err_terminal"].abs().mean(),
                          "model_v1_mae_h": g["err_v1"].abs().mean(), "model_v11_mae_h": g["err_v11"].abs().mean(),
                          "terminal_over_1h": int((g["err_terminal"].abs() > 1).sum()),
                          "model_v1_over_1h": int((g["err_v1"].abs() > 1).sum()),
                          "departure_mae_h": g["err_dep_schedule"].abs().mean()})
    daily = v.groupby("day").apply(agg, include_groups=False).round(3)
    monthly = v.groupby("month").apply(agg, include_groups=False).round(3)
    daily.to_csv(f"{OUT}/daily.csv", encoding="utf-8-sig")
    monthly.to_csv(f"{OUT}/monthly.csv", encoding="utf-8-sig")

    seeds = [round(float(np.abs(v["err_terminal"].values - p).mean()), 3) for p in p_v11]
    summary = {
        "test_period": f"{test[0]['decision_time']:%Y-%m-%d} ~ {test[-1]['decision_time']:%Y-%m-%d}",
        "n_vessels": len(v), "n_days": int(v["day"].nunique()),
        "arrival": {"terminal_plan": err_stats(v["err_terminal"].values), "model_v1": err_stats(v["err_v1"].values),
                    "model_v1.1_seed_mean": err_stats(v["err_v11"].values), "model_v1.1_each_seed_mae": seeds,
                    "baseline_mean_of_same_20": err_stats(v["err_mean20"].values)},
        "days_model_v1_better": int((daily["model_v1_mae_h"] < daily["terminal_mae_h"]).sum()),
        "departure": {"naive_median_dwell": err_stats(v["err_dep_naive"].dropna().values),
                      "schedule_estimator": err_stats(v["err_dep_schedule"].dropna().values)},
        "work_hours": err_stats(v["err_work"].dropna().values),
        "work_complete_time": err_stats(v["err_complete"].dropna().values),
        "note_departure": "엑셀의 출항일시는 완료 선박도 96%가 정각 → 터미널 확정 출항 시각. 실측은 작업완료일시",
        "median_dwell_h_2025_to_2026_04": round(med_dwell, 1),
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(monthly.to_string())


if __name__ == "__main__":
    main()
