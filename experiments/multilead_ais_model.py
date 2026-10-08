"""
리드타임별 확장 모델 실험 틀 (v2.1). AIS와 시점별 HJNC 엑셀이 2~4주 쌓이면 돌린다.

  python experiments/multilead_ais_model.py --snapshots "data/snapshots/*.xlsx" --ais "data/ais/positions_*.csv" \
         --static "data/ais/static_*.csv"

무엇을 하나
  1) 시점별 HJNC 엑셀(스냅샷)에서 "그때의 터미널 예정"을 꺼낸다 (scripts/plan_history.py와 같은 방식)
  2) 가장 최근 엑셀에서 접안이 끝난 배의 실제 접안을 정답으로 붙인다
  3) 각 (배, 스냅샷) 쌍 = 샘플 1개. 리드타임 = 실제 접안 − 스냅샷 시각 → 0-12 / 12-24 / 24-48 / 48-72h 구간
  4) 입력: 그 배의 최근 20시간 AIS [거리, 속력] (20, 2) + 보조 특성(물리 ETA−예정, 선원 입력 ETA−예정, 묘박지 대기 척수)
  5) 비교: 터미널 예정 / 물리 ETA / 모델 B(LSTM, 교수님 구조 그대로) / A+B 혼합 — 리드타임 구간별 MAE
판정 기준 (기획서와 같음): 24·48시간 전 구간에서 터미널 예정보다 MAE가 낮아야 다음 단계(공사 연동 제안)로 간다.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from data.ais_features import load_positions, match_mmsi, scalar_features
from data.hjnc import read_berth_excel

LEAD_BINS = [0, 12, 24, 48, 72]
LEAD_LABELS = ["0-12h", "12-24h", "24-48h", "48-72h"]
DIST_RANGE, SOG_RANGE = (0.0, 1500.0), (0.0, 25.0)     # 고정 범위 스케일 (ArrivalScaler와 같은 원칙)


def snap_time(path):
    import re
    m = re.search(r"(20\d{12})", os.path.basename(path))
    return pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S") if m else None


def build_samples(snap_paths, pos, static):
    snaps = sorted([(snap_time(p), p) for p in snap_paths if snap_time(p) is not None])
    final = read_berth_excel(snaps[-1][1])
    done = final[final["작업 시작일시"].notna()].drop_duplicates("모선항차", keep="last").set_index("모선항차")
    mmsi_of = match_mmsi(static, final)
    st_last = static.sort_values("time_utc").groupby("mmsi").last()
    rows, seqs = [], []
    for t, p in snaps[:-1]:
        df = read_berth_excel(p)
        fut = df[df["작업 시작일시"].isna() & (df["입항일시"] > t)]
        for _, r in fut.iterrows():
            cid = r["모선항차"]
            if cid not in done.index or r["선박명"] not in mmsi_of:
                continue
            actual = done.loc[cid, "입항일시"]
            lead = (actual - t).total_seconds() / 3600
            if not (0 < lead <= LEAD_BINS[-1]):
                continue
            mmsi = mmsi_of[r["선박명"]]
            sf, seq = scalar_features(pos, st_last.loc[mmsi] if mmsi in st_last.index else None, mmsi, t, r["입항일시"])
            if seq is None:
                continue
            rows.append({"call_id": cid, "snapshot": t, "lead_h": lead, "planned_berth": r["입항일시"],
                         "actual_berth": actual, "y": (actual - r["입항일시"]).total_seconds() / 3600, **sf})
            seqs.append(seq)
    s = pd.DataFrame(rows)
    if not s.empty:
        s["lead"] = pd.cut(s["lead_h"], LEAD_BINS, labels=LEAD_LABELS)
    return s, np.array(seqs, dtype="float32") if seqs else np.zeros((0, 20, 2), "float32")


def scale(seqs):
    x = seqs.copy()
    x[..., 0] = (x[..., 0] - DIST_RANGE[0]) / (DIST_RANGE[1] - DIST_RANGE[0])
    x[..., 1] = (x[..., 1] - SOG_RANGE[0]) / (SOG_RANGE[1] - SOG_RANGE[0])
    return np.clip(x, -0.5, 1.5)


def evaluate(s: pd.DataFrame, seqs: np.ndarray, epochs: int = 60):
    from tensorflow import keras
    from serving_app.lstm_model import build_model

    s = s.sort_values("snapshot").reset_index(drop=True)
    cut = s["snapshot"].quantile(0.7)                  # 시간순 70% 학습 / 30% 평가
    tr, te = (s["snapshot"] <= cut).values, (s["snapshot"] > cut).values
    keras.utils.set_random_seed(1)
    m = build_model()                                  # 교수님 구조 그대로, 입력 (20, 2)
    y_s = (s["y"].values + 12) / 24                    # 접안 오차 −12 ~ +12h 고정 범위
    m.fit(scale(seqs[tr]), y_s[tr], epochs=epochs, batch_size=32, verbose=0, validation_split=0.15,
          callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    s["pred_B"] = m.predict(scale(seqs), verbose=0).flatten() * 24 - 12
    out = []
    for lead, g in s[te].groupby("lead", observed=True):
        phys = g["physics_eta_minus_plan_h"]
        out.append({"lead": lead, "n": len(g),
                    "terminal_mae_h": g["y"].abs().mean(),
                    "physics_eta_mae_h": (g["y"] - phys).abs()[phys.notna()].mean() if phys.notna().any() else np.nan,
                    "model_B_mae_h": (g["y"] - g["pred_B"]).abs().mean()})
    return pd.DataFrame(out).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="data/snapshots/*.xlsx")
    ap.add_argument("--ais", default="data/ais/positions_*.csv")
    ap.add_argument("--static", default="data/ais/static_*.csv")
    a = ap.parse_args()
    snaps, pos_p, sta_p = glob.glob(a.snapshots), glob.glob(a.ais), glob.glob(a.static)
    if len(snaps) < 2 or not pos_p or not sta_p:
        print(f"데이터 부족: 스냅샷 {len(snaps)}개, AIS 위치 파일 {len(pos_p)}개, 정적 파일 {len(sta_p)}개. "
              "README의 '확장 데이터 모으기'를 먼저 하세요.")
        return
    pos = load_positions(pos_p)
    static = pd.concat([pd.read_csv(p) for p in sta_p], ignore_index=True)
    s, seqs = build_samples(snaps, pos, static)
    print(f"샘플 {len(s)}개 (배 {s['call_id'].nunique() if len(s) else 0}척)")
    if len(s) < 100:
        print("샘플이 100개 미만이라 학습하지 않습니다. 리드타임별 터미널 예정 오차만 출력합니다.")
        if len(s):
            print(s.groupby("lead", observed=True)["y"].agg(n="size", terminal_mae_h=lambda x: x.abs().mean()).round(3))
        return
    print(evaluate(s, seqs).to_string(index=False))


if __name__ == "__main__":
    main()
