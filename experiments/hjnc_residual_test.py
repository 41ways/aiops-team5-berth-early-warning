"""
실제 HJNC 데이터: '기본 예측 + LSTM 보정' 구조 시험.

A. 접안(도착) 시각
   기본 예측 = 터미널 접안 예정 (반입마감 + 10h, 접안 약 10시간 전 예정)
   타깃      = 실제 접안 − 기본 예측 (시간)
   입력 (20,2)= 그 배 접안 예정 시각 이전에 이미 접안한 최근 20척의 [접안 오차, 물량]
B. 작업시간
   기본 예측 = 물량 기반 선형회귀 (학습 구간 적합)
   타깃      = log(실제 작업시간 / 기본 예측)  (곱셈 보정)
   입력 (20,2)= 그 배 접안 전에 작업이 끝난 최근 20척의 [log 보정비, 물량]
분할: 학습 < 2026-01-01 / 검증 2026-01-01~04-30 / 테스트 2026-05-01~
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from tensorflow import keras

from data.hjnc import completed_calls, read_berth_excel

SEQ = 20
TR_END, VA_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-01")


def lstm(seed=42):
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([keras.layers.Input(shape=(SEQ, 2)), keras.layers.LSTM(32, return_sequences=True),
                          keras.layers.LSTM(32, return_sequences=True), keras.layers.LSTM(16),
                          keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def fit_eval(s, yrange, label):
    tr, va, te = s[s.as_of < TR_END], s[(s.as_of >= TR_END) & (s.as_of < VA_END)], s[s.as_of >= VA_END]
    lo, hi = yrange
    X = lambda df: np.stack(df.seq.values).astype("float32")
    y = lambda df: ((df.target.values - lo) / (hi - lo)).astype("float32")
    m = lstm()
    m.fit(X(tr), y(tr), validation_data=(X(va), y(va)), epochs=100, batch_size=32, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    p = m.predict(X(te), verbose=0).flatten() * (hi - lo) + lo
    return te, p, len(tr), len(va)


def arrival(d):
    d = d.sort_values("입항일시").reset_index(drop=True)
    d["dev"] = (d["입항일시"] - d["plan_berth_10h"]).dt.total_seconds() / 3600
    rows = []
    for i in range(SEQ, len(d)):
        r = d.iloc[i]
        prev = d.iloc[:i]
        prev = prev[prev["입항일시"] <= r["plan_berth_10h"] - pd.Timedelta(hours=10)]  # 예측 시점(접안 10h 전)에 이미 접안한 배만
        if len(prev) < SEQ:
            continue
        h = prev.tail(SEQ)
        seq = np.stack([(h.dev.values + 6) / 12, h.box.values / 8000], 1)
        rows.append({"as_of": r["plan_berth_10h"], "seq": seq, "target": r.dev, "roll": h.dev.mean()})
    s = pd.DataFrame(rows)
    te, p, ntr, nva = fit_eval(s, (-6, 6), "arrival")
    trn = s[s.as_of < TR_END]
    med = trn.target.median()
    out = {"n_train": ntr, "n_val": nva, "n_test": len(te),
           "terminal_plan_MAE_h": np.mean(np.abs(te.target)),
           "plan+train_median_bias_MAE_h": np.mean(np.abs(te.target - med)), "train_median_bias_h": med,
           "plan+last20_mean_MAE_h": np.mean(np.abs(te.target - te.roll)),
           "LSTM_correction_MAE_h": np.mean(np.abs(te.target - p)),
           "share_abs_err_gt1h_plan": np.mean(np.abs(te.target) > 1), "share_abs_err_gt1h_LSTM": np.mean(np.abs(te.target - p) > 1)}
    print("A arrival", {k: round(float(v), 3) for k, v in out.items()}, flush=True)


def worktime(d):
    tr0 = d[d["입항일시"] < TR_END]
    A = np.stack([np.ones(len(tr0)), tr0.box.values], 1)
    w, *_ = np.linalg.lstsq(A, tr0.work_h.values, rcond=None)
    d = d.copy()
    d["base"] = np.clip(w[0] + w[1] * d.box.values, 1.0, None)
    d["lr"] = np.log(d.work_h / d.base)
    rows = []
    for i, r in d.iterrows():
        prev = d[d["작업 완료일시"] < r["입항일시"]]
        if len(prev) < SEQ:
            continue
        h = prev.tail(SEQ)
        seq = np.stack([(h.lr.values + 1.5) / 3, h.box.values / 8000], 1)
        rows.append({"as_of": r["입항일시"], "seq": seq, "target": r.lr, "base": r.base, "work_h": r.work_h,
                     "roll": h.lr.mean()})
    s = pd.DataFrame(rows)
    te, p, ntr, nva = fit_eval(s, (-1.5, 1.5), "worktime")
    out = {"n_train": ntr, "n_val": nva, "n_test": len(te), "linear_a": w[0], "linear_b_per_box": w[1],
           "linear_MAE_h": np.mean(np.abs(te.base - te.work_h)),
           "linear*last20_ratio_MAE_h": np.mean(np.abs(te.base * np.exp(te.roll) - te.work_h)),
           "linear*LSTM_MAE_h": np.mean(np.abs(te.base * np.exp(p) - te.work_h)),
           "test_work_h_mean": te.work_h.mean()}
    print("B worktime", {k: round(float(v), 4) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    d = completed_calls(read_berth_excel(sys.argv[1]))
    arrival(d)
    worktime(d)
