"""
HJNC 선석 스케줄 → LSTM 입력 시퀀스 (학습·서빙·재학습 공용).

교수님 예제(HAIC)와 같은 역할: 시퀀스 정의를 이 파일 한 곳에서만 관리해야
"학습 시점 입력"과 "서빙 시점 입력"이 어긋나지 않는다 (빈칸 2 Training-serving skew).

예제와의 대응
  HAIC                       | 이 모델 (부산항 HJNC 접안 시각 보정)
  최근 20거래일              | 같은 항로의 최근 20척 (모자라면 터미널 전체 최근 배로 채움)
  피처 close                 | 접안 오차(h) = 실제 접안 − 터미널 접안 예정
  피처 volume                | 처리 물량(box) = 양하 + 선적 + S/H
  타깃: 다음날 종가          | 타깃: 다음 배의 접안 오차(h) → 예측 접안 = 터미널 예정 + 예측 오차

터미널 접안 예정 = '반입 마감 + 10시간' (반입 마감은 접안 예정 10시간 전으로 고정되는 규칙, 데이터로 확인)
예측 시점        = 반입 마감 시각 (접안 약 10시간 전). 그 시각에 이미 접안한 배만 입력으로 쓴다 → 미래 정보 누수 없음.
"""
import pickle

import numpy as np
import pandas as pd

from data.hjnc import completed_calls, read_berth_excel

SEQ_LEN = 20
GROUP_KEY = "항로"
LEAD_H = 10.0

TRAIN_END = pd.Timestamp("2026-01-01")   # 학습: 이 날짜 전 (2025-01 ~ 2025-12)
VAL_END = pd.Timestamp("2026-05-01")     # 검증(게이트): 2026-01 ~ 04 / 테스트·운영 리플레이: 2026-05 ~


def _dev(df: pd.DataFrame) -> pd.Series:
    return (df["입항일시"] - df["plan_berth_10h"]).dt.total_seconds() / 3600


def berthed_history(df: pd.DataFrame) -> pd.DataFrame:
    """접안이 끝나 실제 접안 시각을 아는 배 (작업 완료 + 작업 중). 접안 순 정렬."""
    h = df[df["작업 시작일시"].notna()].copy()
    h["dev"] = _dev(h)
    h = h[h["dev"].abs() <= 48]  # 기록 오류 방어
    return h.sort_values("입항일시").reset_index(drop=True)


def _history_for(hist: pd.DataFrame, group_value, as_of: pd.Timestamp) -> pd.DataFrame | None:
    prev = hist[hist["입항일시"] <= as_of]
    if len(prev) < SEQ_LEN:
        return None
    g = prev[prev[GROUP_KEY] == group_value]
    if len(g) >= SEQ_LEN:
        return g.tail(SEQ_LEN)
    return pd.concat([prev[prev[GROUP_KEY] != group_value].tail(SEQ_LEN - len(g)), g])


def sequence_points(h: pd.DataFrame) -> list[dict]:
    return [{"delay_hours": float(a), "box": float(b)} for a, b in zip(h["dev"].values, h["box"].values)]


def load_samples(path: str) -> list[dict]:
    """완료된 배 1척 = 샘플 1개. 예측 시점(반입 마감) 순으로 정렬."""
    df = read_berth_excel(path)
    hist = berthed_history(df)
    done = completed_calls(df)
    done = done.assign(dev=_dev(done))
    done = done[done["dev"].abs() <= 48].sort_values("입항일시")
    samples = []
    for _, r in done.iterrows():
        as_of = r["plan_berth_10h"] - pd.Timedelta(hours=LEAD_H)
        h = _history_for(hist, r[GROUP_KEY], as_of)
        if h is None:
            continue
        samples.append({"call_id": r["모선항차"], "vessel": r["선박명"], "route": r[GROUP_KEY], "berth": r["선석"],
                        "decision_time": as_of, "planned_berth": r["plan_berth_10h"], "actual_berth": r["입항일시"],
                        "sequence": sequence_points(h), "actual_correction": float(r["dev"])})
    samples.sort(key=lambda s: s["decision_time"])
    return samples


def upcoming_requests(path: str, as_of: pd.Timestamp, horizon_h: float = 30.0) -> list[dict]:
    """as_of 시점 기준 아직 접안하지 않은 배 중 horizon_h 안에 접안 예정인 배의 예측 요청."""
    df = read_berth_excel(path)
    hist = berthed_history(df)
    fut = df[df["작업 시작일시"].isna() & (df["입항일시"] > as_of) &
             (df["입항일시"] <= as_of + pd.Timedelta(hours=horizon_h))].sort_values("입항일시")
    out = []
    for _, r in fut.iterrows():
        h = _history_for(hist, r[GROUP_KEY], as_of)
        if h is None:
            continue
        out.append({"row": r, "sequence": sequence_points(h)})
    return out


class ArrivalScaler:
    """
    접안 오차·물량을 [0,1] 근처로. 데이터로 fit하지 않고 업무 범위로 고정한다
    (혼잡기에 학습 최대값을 넘는 지연이 와도 스케일이 깨지지 않게). 재학습 때 다시 맞추지 않는 원칙은 예제와 같다.
    """

    def __init__(self, delay=(-6.0, 6.0), box=(0.0, 8000.0)):
        self.delay, self.box = delay, box

    def fit(self, *_):
        return self

    @staticmethod
    def _s(v, lo, hi):
        return (v - lo) / (hi - lo)

    def transform_point(self, delay_hours: float, box: float) -> list[float]:
        return [self._s(delay_hours, *self.delay), self._s(box, *self.box)]

    def scale_target(self, delay_hours: float) -> float:
        return self._s(delay_hours, *self.delay)

    def inverse_target(self, scaled: float) -> float:
        lo, hi = self.delay
        return scaled * (hi - lo) + lo

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "ArrivalScaler":
        s = cls()
        with open(path, "rb") as f:
            s.__dict__.update(pickle.load(f))
        return s


def build_sequences(samples: list[dict], scaler: ArrivalScaler):
    X = np.array([[scaler.transform_point(p["delay_hours"], p["box"]) for p in s["sequence"]] for s in samples],
                 dtype="float32")
    y = [s["actual_correction"] for s in samples]
    return X, y


def split_by_period(samples: list[dict]):
    tr = [s for s in samples if s["decision_time"] < TRAIN_END]
    va = [s for s in samples if TRAIN_END <= s["decision_time"] < VAL_END]
    te = [s for s in samples if s["decision_time"] >= VAL_END]
    return tr, va, te


def train_test_split(X, y, test_ratio: float = 0.2):
    """시간 순서를 유지한 채 앞은 train, 뒤는 test (재학습 시 최근 샘플 안에서 사용)."""
    i = int(len(X) * (1 - test_ratio))
    return X[:i], y[:i], X[i:], y[i:]
