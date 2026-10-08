"""
로컬 baseline 모델 (MLflow 없이, 예제 Day1 사전 준비).

  serving_app/models/scaler.pkl        ← 업무 범위로 고정된 스케일러
  serving_app/models/arrival_v1.keras  ← 학습된 LSTM

실행: python scripts/train_baseline_v1.py [엑셀 경로]   (경로를 안 주면 data/uploads/의 최신 파일)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import ArrivalScaler, load_samples, split_by_period
from data.storage import latest_upload
from serving_app.train_and_register import fit_base, metrics, predict

MODEL_PATH = "serving_app/models/arrival_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
BAND_PATH = "serving_app/models/residual_band.json"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else latest_upload()
    samples = load_samples(path)
    train, val, test = split_by_period(samples)
    print(f"samples: train {len(train)} / val {len(val)} / test {len(test)}")

    scaler = ArrivalScaler().fit(train)
    scaler.save(SCALER_PATH)
    model = fit_base(train, scaler)  # 빈칸 1: X_train, y_train_scaled 로만 학습 (검증·테스트는 학습에 안 씀)

    for name, part in (("val", val), ("test", test)):
        m = metrics([s["actual_correction"] for s in part], predict(model, part, scaler))
        print(f"{name}: model MAE {m['mae']:.3f}h | 터미널 예정 MAE {m['baseline_mae']:.3f}h | RMSE {m['rmse']:.3f}h")

    # v1.2: 예측 범위. 검증 구간 오차(실제 − 예측)의 10%·90% 분위 → 예측 접안 ± 이 값이 80% 범위
    import json
    import numpy as np
    r = np.array([s["actual_correction"] for s in val]) - np.array(predict(model, val, scaler))
    band = {"q10": round(float(np.quantile(r, 0.1)), 3), "q90": round(float(np.quantile(r, 0.9)), 3), "from": "val 2026-01~04"}

    # v1.4: 작업시간(→ 출항) 80% 범위. 검증 구간에서 실제 ÷ 예측 작업시간의 10%·90% 분위 (곱하는 배수)
    import pandas as pd
    from data.features import TRAIN_END, VAL_END
    from data.hjnc import completed_calls, read_berth_excel
    from serving_app.schedule import WorkTimeEstimator
    df = read_berth_excel(path)
    est = WorkTimeEstimator(df, fit_before=VAL_END)
    d = completed_calls(df)
    d = d[(d["입항일시"] >= TRAIN_END) & (d["입항일시"] < VAL_END)]
    pw = np.array([est.work_hours(b, rt, t - pd.Timedelta(hours=10))
                   for b, rt, t in zip(d["box"], d["항로"], d["plan_berth_10h"])])
    ratio = np.log(d["work_h"].values / pw)
    band.update(work_x10=round(float(np.exp(np.quantile(ratio, 0.1))), 3),
                work_x90=round(float(np.exp(np.quantile(ratio, 0.9))), 3))
    json.dump(band, open(BAND_PATH, "w"))
    print(f"80% band: {band}")
    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
