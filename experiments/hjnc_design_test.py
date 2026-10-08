"""
실제 HJNC 데이터: 입력 시퀀스 구성별로 '다음 배 작업시간' 예측 성능 비교.

공통: 예측 대상 배의 접안 시각(as_of) 이전에 작업이 끝난 배만 입력으로 쓴다 (미래 누수 없음).
타깃: 그 배의 생산성(box/h). 작업시간 예측 = 그 배 물량 ÷ 예측 생산성.
분할: 학습 as_of < 2026-01-01 / 검증 2026-01-01~04-30 / 테스트 2026-05-01~
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
PROD_RANGE, BOX_RANGE = (0.0, 200.0), (0.0, 8000.0)


def build(d: pd.DataFrame, key: str | None):
    """key=None: 터미널 전체 최근 20척 / key='항로' 등: 같은 그룹 최근 20척 (모자라면 터미널 전체로 채움)."""
    ends = d["작업 완료일시"].values
    rows = []
    for i, r in d.iterrows():
        as_of = r["입항일시"]
        prev = d[d["작업 완료일시"] < as_of]
        if len(prev) < SEQ:
            continue
        hist = prev
        if key:
            g = prev[prev[key] == r[key]]
            hist = pd.concat([prev[prev[key] != r[key]].tail(max(0, SEQ - len(g))), g]).sort_values("작업 완료일시") if len(g) < SEQ else g
        h = hist.tail(SEQ)
        seq = np.stack([(h["prod"].values - PROD_RANGE[0]) / (PROD_RANGE[1] - PROD_RANGE[0]),
                        (h["box"].values - BOX_RANGE[0]) / (BOX_RANGE[1] - BOX_RANGE[0])], axis=1)
        rows.append({"as_of": as_of, "seq": seq, "prod": r["prod"], "box": r["box"], "work_h": r["work_h"],
                     "naive": h["prod"].mean(), "same_route_n": int((prev["항로"] == r["항로"]).sum()),
                     "route_mean": prev.loc[prev["항로"] == r["항로"], "prod"].tail(10).mean(),
                     "line": r["선사"], "route": r["항로"]})
    return pd.DataFrame(rows)


def lstm(seed=42):
    keras.utils.set_random_seed(seed)
    m = keras.Sequential([keras.layers.Input(shape=(SEQ, 2)), keras.layers.LSTM(32, return_sequences=True),
                          keras.layers.LSTM(32, return_sequences=True), keras.layers.LSTM(16),
                          keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def wh_mae(box, prod_pred, work_h):
    return float(np.mean(np.abs(box / np.clip(prod_pred, 5, None) - work_h)))


def main(path):
    d = completed_calls(read_berth_excel(path))
    print("completed calls used:", len(d))
    tr_end, va_end = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-05-01")
    res = {}
    for name, key in (("terminal20", None), ("route20", "항로"), ("berth20", "선석")):
        s = build(d, key)
        tr, va, te = s[s.as_of < tr_end], s[(s.as_of >= tr_end) & (s.as_of < va_end)], s[s.as_of >= va_end]
        X = lambda df: np.stack(df.seq.values).astype("float32")
        y = lambda df: ((df["prod"].values - PROD_RANGE[0]) / (PROD_RANGE[1] - PROD_RANGE[0])).astype("float32")
        m = lstm()
        es = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        m.fit(X(tr), y(tr), validation_data=(X(va), y(va)), epochs=100, batch_size=32, verbose=0, callbacks=[es])
        p = m.predict(X(te), verbose=0).flatten() * (PROD_RANGE[1] - PROD_RANGE[0]) + PROD_RANGE[0]
        # 선형회귀 (작업시간 = a + b·물량), 학습 구간으로 적합
        A = np.stack([np.ones(len(tr)), tr.box.values], 1)
        w, *_ = np.linalg.lstsq(A, tr.work_h.values, rcond=None)
        lin = w[0] + w[1] * te.box.values
        rm = te.route_mean.fillna(te.naive)
        res[name] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                     "LSTM": wh_mae(te.box.values, p, te.work_h.values),
                     "naive_terminal20_mean": wh_mae(te.box.values, te.naive.values, te.work_h.values),
                     "route_last10_mean": wh_mae(te.box.values, rm.values, te.work_h.values),
                     "linear_box": float(np.mean(np.abs(lin - te.work_h.values))),
                     "test_work_h_mean": float(te.work_h.mean()), "epochs": len(m.history.history["loss"])}
        print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
