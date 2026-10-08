"""
드리프트 감지 (최근 21척).

  relative : 모델 MAE ≥ 터미널 예정 MAE + 0.05h(3분)  → 기존 예측보다 못해짐
             v1.1: 여유 0.05h 추가. 여유 0이면 테스트 구간 21척 창의 6.6%가 우연히 걸렸다 → 0.05h면 1.2%
             (reports/backtest_v1.0/vessels.csv 로 계산)
  absolute : 모델 RMSE > 1.0시간          → 예제의 'RMSE > $4' 자리. 평소 오차(약 0.2~0.3h)의 3~4배
  hybrid(기본) = relative 또는 absolute

recent_predictions 한 건: {"predicted": 예측 오차, "actual": 실제 오차, "baseline": 0.0 (터미널 예정 그대로)}
"""
import math
import os

WINDOW_SIZE = 21
ABS_RMSE_THRESHOLD = 1.0
REL_TOLERANCE_HOURS = float(os.getenv("REL_TOLERANCE_HOURS", "0.05"))


def compute_rmse(recent: list[dict], key: str = "predicted") -> float:
    if not recent:
        return 0.0
    errors_sq = [(p["actual"] - p[key]) ** 2 for p in recent]   # 빈칸 7: 실제 − 예측
    return math.sqrt(sum(errors_sq) / len(errors_sq))


def compute_mae(recent: list[dict], key: str = "predicted") -> float:
    if not recent:
        return 0.0
    return sum(abs(p["actual"] - p[key]) for p in recent) / len(recent)


def drift_reasons(recent_predictions: list[dict]) -> dict:
    if len(recent_predictions) < WINDOW_SIZE:   # 빈칸 8: 충분히 쌓이기 전에는 판단하지 않음
        return {"relative": False, "absolute": False}
    w = recent_predictions[-WINDOW_SIZE:]
    return {"relative": compute_mae(w, "predicted") - compute_mae(w, "baseline") >= REL_TOLERANCE_HOURS,
            "absolute": compute_rmse(w) > ABS_RMSE_THRESHOLD}


def is_drift(recent_predictions: list[dict], mode: str | None = None) -> bool:
    mode = mode or os.getenv("DRIFT_MODE", "hybrid")
    r = drift_reasons(recent_predictions)
    return {"relative": r["relative"], "absolute": r["absolute"]}.get(mode, r["relative"] or r["absolute"])
