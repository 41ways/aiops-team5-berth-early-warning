"""
예측 API.

   POST /predict             : 같은 항로 최근 20척 이력 → 다음 배의 접안 오차 (+ 예정을 주면 예측 접안 시각)
   POST /predict/batch-test  : 정답이 확정된 배 여러 척 → 연속 예측 → 드리프트 검사 (예제와 같은 역할)
   POST /predict/replay      : 업로드된 스케줄에서 기간을 골라 batch-test로 흘려보냄 (예제의 simulate_drift 배치 자리)
                               inject_shift_h 를 주면 정답에 인위적 지연을 더해 드리프트를 재현 (예제의 '변동성 3배' 자리)
   GET  /predict/next-day    : as_of 시점 기준 다음 30시간 안에 접안할 배의 예측 접안·예측 출항 (일정표용)
"""
import random

import pandas as pd
from fastapi import APIRouter, HTTPException

from serving_app import model_loader, state
from serving_app.monitoring.drift_detector import WINDOW_SIZE, compute_mae
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import BatchTestRequest, BatchTestResponse, CaseIn, PredictRequest, PredictResponse

router = APIRouter()


def mean20(seq: list[dict]) -> float:
    """학습 없는 보정: 같은 항로 최근 20척 접안 오차의 평균 (v2.2 폴백 1단계)."""
    return sum(p["delay_hours"] for p in seq) / len(seq)


def fallback_value(seq: list[dict]) -> float:
    return mean20(seq) if state.fallback_target == "mean20" else 0.0


def fallback_label() -> str:
    return "fallback:route-mean20" if state.fallback_target == "mean20" else "fallback:terminal-plan"

recent_predictions: list[dict] = []

REPLAY_PERIODS = {
    "normal": ("2026-05-01", "2026-07-31"),
    "busy": ("2026-08-01", "2026-10-08"),   # 실제 데이터에서 선석 체류시간이 18h → 27h로 늘어난 구간
}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = model_loader.get_model()
    delay = model.predict_one([p.model_dump() for p in req.sequence])
    version = model.version
    if state.fallback:
        delay, version = fallback_value([p.model_dump() for p in req.sequence]), fallback_label()
    berth = None
    if req.planned_berth:
        try:
            berth = (pd.Timestamp(req.planned_berth) + pd.Timedelta(hours=delay)).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            raise HTTPException(422, "planned_berth 형식은 'YYYY-MM-DD HH:MM'")
    return PredictResponse(predicted_delay_hours=round(delay, 3), predicted_berth=berth, model_version=version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    model = model_loader.get_model()
    preds = model.predict_many([[p.model_dump() for p in c.sequence] for c in req.cases])
    seqs = [[p.model_dump() for p in c.sequence] for c in req.cases]
    served = [fallback_value(sq) for sq in seqs] if state.fallback else preds
    batch = []
    for c, sq, pred, out in zip(req.cases, seqs, preds, served):
        rec = {"predicted": pred, "actual": c.actual_delay_hours, "baseline": 0.0, "mean20": mean20(sq),
               "decision_time": c.decision_time}
        batch.append({**rec, "served": out})
        recent_predictions.append(rec)
    recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]

    drift_check = check_and_trigger(recent_predictions)
    if drift_check.get("status") in ("retrain_triggered", "retrain_scheduled"):
        recent_predictions.clear()
    return BatchTestResponse(predictions=[round(p, 3) for p in served],
                             actuals=[c.actual_delay_hours for c in req.cases],
                             model_mae=round(compute_mae(batch, "served"), 3),
                             baseline_mae=round(compute_mae(batch, "baseline"), 3),
                             drift_check=drift_check)


@router.post("/predict/replay", response_model=BatchTestResponse)
def replay(period: str = "normal", start: str | None = None, end: str | None = None,
           n: int = WINDOW_SIZE, inject_shift_h: float = 0.0):
    from data.features import load_samples
    from data.storage import latest_upload

    if not (start and end):
        if period not in REPLAY_PERIODS:
            raise HTTPException(422, f"period는 {list(REPLAY_PERIODS)} 중 하나이거나 start/end를 주세요")
        start, end = REPLAY_PERIODS[period]
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    pool = [s for s in load_samples(latest_upload()) if lo <= s["decision_time"] < hi]
    if len(pool) < n:
        raise HTTPException(422, f"{start}~{end} 기간의 정답 있는 배가 {len(pool)}척뿐입니다 (필요 {n})")
    state.sim_shift_h = inject_shift_h
    i = random.randint(0, len(pool) - n)
    cases = [CaseIn(sequence=s["sequence"], actual_delay_hours=max(-48, min(48, s["actual_correction"] + inject_shift_h)),
                    decision_time=s["decision_time"].isoformat()) for s in pool[i:i + n]]
    return batch_test(BatchTestRequest(cases=cases))


@router.get("/predict/next-day")
def next_day(as_of: str | None = None, horizon_h: float = 30.0):
    from data.features import upcoming_requests
    from data.hjnc import read_berth_excel
    from data.storage import latest_upload
    from serving_app.schedule import WorkTimeEstimator

    path = latest_upload()
    t0 = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().floor("min")
    reqs = upcoming_requests(path, t0, horizon_h)
    if not reqs:
        return {"as_of": str(t0), "vessels": []}
    model = model_loader.get_model()
    delays = [fallback_value(r["sequence"]) for r in reqs] if state.fallback else model.predict_many([r["sequence"] for r in reqs])
    est = WorkTimeEstimator(read_berth_excel(path))
    import json
    import os
    bp = "serving_app/models/residual_band.json"
    band = json.load(open(bp)) if os.path.exists(bp) else {"q10": -0.31, "q90": 0.33}
    out = []
    for r, d in zip(reqs, delays):
        row = r["row"]
        berth = row["입항일시"] + pd.Timedelta(hours=d)
        dep = est.departure(berth, row["box"], row["항로"], t0)
        out.append({"call_id": row["모선항차"], "vessel": row["선박명"], "berth_no": row["선석"], "line": row["선사"],
                    "route": row["항로"], "box": int(row["box"]),
                    "terminal_planned_berth": row["입항일시"].strftime("%Y-%m-%d %H:%M"),
                    "predicted_berth": berth.strftime("%Y-%m-%d %H:%M"), "predicted_delay_h": round(d, 2),
                    "predicted_berth_80": [(berth + pd.Timedelta(hours=band["q10"])).strftime("%H:%M"),
                                           (berth + pd.Timedelta(hours=band["q90"])).strftime("%H:%M")],
                    "terminal_planned_departure": row["출항일시"].strftime("%Y-%m-%d %H:%M"),
                    "predicted_departure": dep.strftime("%Y-%m-%d %H:%M"),
                    "predicted_departure_80": [est.departure(berth, row["box"], row["항로"], t0, band.get("work_x10", 0.78)).strftime("%m-%d %H:%M"),
                                               est.departure(berth, row["box"], row["항로"], t0, band.get("work_x90", 1.32)).strftime("%m-%d %H:%M")],
                    "predicted_work_h": round(est.work_hours(row["box"], row["항로"], t0), 1)})
    return {"as_of": str(t0), "model_version": fallback_label() if state.fallback else model.version,
            "vessels": out}
