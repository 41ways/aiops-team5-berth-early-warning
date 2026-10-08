"""
v2.5: 드리프트 판정 = 최근 14일(배 척수 아님) + 평균 치우침 규칙, 재학습 = 최근 28일.

교수님 예제의 21(주가 영업일 약 한 달)을 그대로 쓰던 것을 항만 단위로 바꾼다. 값은 addons/rules.py.
  판정 창   : 최근 drift_window_days(14)일 안에 정답이 확정된 배. drift_min_ships(15)척 미만이면 판정 보류
  드리프트  : relative (모델 MAE ≥ 터미널 예정 MAE + 3분)  또는  absolute (RMSE > 60분)
              또는  bias (|평균(실제 − 모델)| ≥ 20분)  ← 모두 같이 늦어지는 변화용
  재학습    : 최근 retrain_days(28)일 정답 확정 배로 fine-tuning (기존 retrain_trigger 그대로, 기간만)

기존 serving_app 파일은 고치지 않는다. 이 모듈을 import 하면
  - retrain_trigger 가 쓰는 drift_reasons / is_drift / RETRAIN_DAYS 를 바꿔 끼우고
  - predict.batch_test 가 기록을 21건만 남기던 것을 넉넉히(시간으로 다시 거름) 바꾸고
  - POST /predict/replay 를 "터미널 하나의 14일치"를 보내는 버전으로 앞에 끼운다
로그 문장([WARN] drift detected 등)은 그대로라 대시보드·관리자 콘솔이 읽는 방식은 같다.
"""
import random

import pandas as pd
from fastapi import HTTPException
from fastapi.routing import APIRoute

from addons.rules import RULES
from serving_app.main import app
from serving_app.monitoring import drift_detector as dd
from serving_app.monitoring import retrain_trigger as rt
from serving_app.routers import predict as pr
from serving_app.schemas import BatchTestRequest, CaseIn

WINDOW_DAYS = RULES["drift_window_days"]["value"]
MIN_SHIPS = RULES["drift_min_ships"]["value"]
BIAS_H = RULES["drift_bias_h"]["value"]
RETRAIN_DAYS = RULES["retrain_days"]["value"]
KEEP = 500  # batch_test 가 남기는 최근 기록 수 (판정은 아래에서 14일로 다시 거른다)


def stats(pairs: list[tuple[float, float]]) -> dict:
    """pairs = [(실제, 예측)] (이미 14일 창으로 거른 것) → 지표와 드리프트 이유."""
    n = len(pairs)
    if not n:
        return {"n": 0, "full": False, "drift": False, "relative": False, "absolute": False, "bias_rule": False,
                "window_days": WINDOW_DAYS, "min_ships": MIN_SHIPS}
    mae = sum(abs(a - p) for a, p in pairs) / n
    term = sum(abs(a) for a, _ in pairs) / n
    rmse = (sum((a - p) ** 2 for a, p in pairs) / n) ** 0.5
    bias = sum(a - p for a, p in pairs) / n
    full = n >= MIN_SHIPS
    rel = full and mae - term >= dd.REL_TOLERANCE_HOURS
    ab = full and rmse > dd.ABS_RMSE_THRESHOLD
    bi = full and abs(bias) >= BIAS_H
    return {"n": n, "full": full, "mae": round(mae, 3), "term_mae": round(term, 3), "rmse": round(rmse, 3), "bias": round(bias, 3),
            "in30": round(sum(abs(a - p) <= 0.5 for a, p in pairs) / n, 3),
            "term_in30": round(sum(abs(a) <= 0.5 for a, _ in pairs) / n, 3),
            "drift": bool(rel or ab or bi), "relative": bool(rel), "absolute": bool(ab), "bias_rule": bool(bi),
            "window_days": WINDOW_DAYS, "min_ships": MIN_SHIPS}


def in_window(records: list[dict], time_key: str = "decision_time") -> list[dict]:
    """가장 최근 기록 시각 기준 14일 안의 기록만."""
    ts = [pd.Timestamp(r[time_key]) for r in records if r.get(time_key)]
    if not ts:
        return records
    lo = max(ts) - pd.Timedelta(days=WINDOW_DAYS)
    return [r for r in records if r.get(time_key) and pd.Timestamp(r[time_key]) > lo]


def drift_reasons(recent: list[dict]) -> dict:
    s = stats([(p["actual"], p["predicted"]) for p in in_window(recent)])
    return {"relative": s["relative"], "absolute": s["absolute"], "bias": s["bias_rule"], "n": s["n"], "window_days": WINDOW_DAYS}


def is_drift(recent: list[dict], mode: str | None = None) -> bool:
    r = drift_reasons(recent)
    return bool(r["relative"] or r["absolute"] or r["bias"])


# 기존 모듈에 끼워 넣기 (파일은 그대로)
rt.drift_reasons = drift_reasons
rt.is_drift = is_drift
rt.RETRAIN_DAYS = RETRAIN_DAYS
pr.WINDOW_SIZE = KEEP

REPLAY_PERIODS = {"normal": ("2026-05-01", "2026-07-31"), "busy": ("2026-08-01", "2026-10-08")}


def replay(period: str = "normal", terminal: str = "HJNC", start: str | None = None, end: str | None = None,
           days: int = WINDOW_DAYS, inject_shift_h: float = 0.0):
    """터미널 하나의 연속 days(기본 14)일치 정답 확정 배를 batch-test 로 흘려보낸다. inject_shift_h 는 정답에 더할 인위 지연."""
    from addons import dashboard as dv
    from data.features import load_samples
    from data.storage import latest_upload

    parts = dv._split(latest_upload())
    if terminal not in parts or not dv._labelable(parts[terminal][0]):
        raise HTTPException(422, f"terminal 은 예측 가능한 터미널 중 하나: {[t for t, (d, _) in parts.items() if dv._labelable(d)]}")
    if not (start and end):
        if period not in REPLAY_PERIODS:
            raise HTTPException(422, f"period 는 {list(REPLAY_PERIODS)} 중 하나이거나 start/end 를 주세요")
        start, end = REPLAY_PERIODS[period]
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    pool = [s for s in load_samples(parts[terminal][1]) if lo <= s["decision_time"] < hi]
    if not pool:
        raise HTTPException(422, f"{terminal} {start}~{end} 에 정답 있는 배가 없습니다")
    span = pd.Timedelta(days=days)
    first, last = pool[0]["decision_time"], pool[-1]["decision_time"]
    if last - first < span:
        t0 = first
    else:
        t0 = first + (last - first - span) * random.random()
    cases_s = [s for s in pool if t0 <= s["decision_time"] < t0 + span]
    if len(cases_s) < MIN_SHIPS:
        raise HTTPException(422, f"{terminal} {days}일치가 {len(cases_s)}척뿐입니다 (판정 최소 {MIN_SHIPS}척)")
    pr.recent_predictions.clear()  # 시연 배치끼리 섞이지 않게: 이번 14일치로만 판정
    rt.state.sim_shift_h = inject_shift_h
    cases = [CaseIn(sequence=s["sequence"], actual_delay_hours=max(-48, min(48, s["actual_correction"] + inject_shift_h)),
                    decision_time=s["decision_time"].isoformat()) for s in cases_s]
    out = pr.batch_test(BatchTestRequest(cases=cases))
    d = out.model_dump() if hasattr(out, "model_dump") else dict(out)
    d["replay"] = {"terminal": terminal, "from": f"{t0:%Y-%m-%d %H:%M}", "to": f"{t0 + span:%Y-%m-%d %H:%M}", "n_ships": len(cases),
                   "inject_shift_h": inject_shift_h, "window_days": days, "min_ships": MIN_SHIPS,
                   "suggested_shifts_h": [0.5, 1.0], "note": "+2시간은 평소 오차의 약 8배라 현실성이 낮음. +30분·+1시간 권장"}
    return d


app.router.routes.insert(0, APIRoute("/predict/replay", replay, methods=["POST"]))


# ---------- v2.5 승격 정책: 확실히 좋으면 자동 승격, 별로면 이전 버전 자동 유지 ----------
# 확실히 좋음 = 게이트 통과(운영 모델 이하, 60분 이하) + 터미널 예정보다 좋음 + 운영 모델보다 1분 이상 좋음 → Production
# 그 밖(게이트는 넘었지만 차이가 작음) → 후보(Staging)로만 등록, Production 은 이전 버전 그대로. 관리 콘솔에서 수동 승격 가능
#   POST /admin/api/models/{version}/promote
# 게이트 실패 → 이전 버전 유지 (+ 터미널 예정보다 못하면 폴백). 예전 버전 되돌리기도 같은 기준(확실히 좋을 때만 자동)
import logging  # noqa: E402
import os  # noqa: E402

from serving_app import model_loader, state  # noqa: E402
from serving_app import train_and_register as tr  # noqa: E402

# AUTO_PROMOTE: margin(기본) = 성능이 확실히 좋으면 자동 승격, 아니면 후보로만 / 1 = 게이트만 넘으면 자동 / 0 = 항상 수동
PROMOTE_MODE = os.getenv("AUTO_PROMOTE", "margin")
AUTO_PROMOTE = PROMOTE_MODE == "1"
MARGIN_H = RULES["promote_margin_h"]["value"]
_orig_register = tr._register_if_gate_passed
_log = logging.getLogger("aiops")


def _register_staging(run_id: str, m: dict, gate=tr._gate) -> dict:
    if AUTO_PROMOTE or gate is not tr._gate_fine_tune:
        return _orig_register(run_id, m, gate)
    result = {"run_id": run_id, **m, "promoted": False, "staged": False}
    clearly_better = (gate(m) and m["mae"] < m["baseline_mae"]
                      and m.get("prod_mae", float("inf")) - m["mae"] >= MARGIN_H)
    if PROMOTE_MODE == "margin" and clearly_better:
        print(f"[AUTO PROMOTE] mae={m['mae']:.3f}h, 운영 모델보다 {(m['prod_mae'] - m['mae']) * 60:.1f}분 좋고 터미널 예정({m['baseline_mae']:.3f}h)보다 좋음")
        return {**_orig_register(run_id, m, gate), "auto_reason": "margin"}
    if gate(m):
        v = tr.mlflow.register_model(f"runs:/{run_id}/model", tr.MODEL_NAME)
        tr.MlflowClient().transition_model_version_stage(name=tr.MODEL_NAME, version=v.version, stage="Staging",
                                                         archive_existing_versions=False)
        result.update(staged=True, version=v.version)
        print(f"[GATE PASSED] mae={m['mae']:.3f}h (current prod {m.get('prod_mae', float('nan')):.3f}h) -> "
              f"{tr.MODEL_NAME} v{v.version} Staging (수동 승격 대기, Production 유지)")
    else:
        print(f"[GATE FAILED] mae={m['mae']:.3f}h -> 배포 차단, 기존 Production 유지")
    return result


tr._register_if_gate_passed = _register_staging


def _retrain_inner(times: list, reasons: dict, sim_shift_h: float, window: list) -> dict:
    """retrain_trigger._retrain_inner 와 같고, 후보 등록(staged)일 때 Production 을 그대로 두는 분기만 추가."""
    import copy
    from datetime import timedelta

    from data.features import load_samples
    from data.storage import latest_upload

    samples = load_samples(latest_upload())
    now = max(times) if times else samples[-1]["decision_time"]
    recent = [s for s in samples if now - timedelta(days=rt.RETRAIN_DAYS) <= s["decision_time"] < now and s["actual_berth"] <= now]
    if sim_shift_h:
        recent = copy.deepcopy(recent)
        for s in recent:
            s["actual_correction"] += sim_shift_h
            for pt in s["sequence"]:
                pt["delay_hours"] += sim_shift_h
        _log.info(f"[INFO] simulation: +{sim_shift_h}h shift applied to retrain window")
    if len(recent) < 10:
        _log.info(f"[INFO] too few labeled calls for retrain ({len(recent)}) - keep current production")
        return {"status": "retrain_skipped", "n_cases": len(recent), "reasons": reasons, "fallback": state.fallback}

    prod_before = model_loader.production_version()
    result = tr.fine_tune(recent)
    rb_clear = (result.get("rollback") and result["mae"] < result["baseline_mae"]
                and result.get("prod_mae", float("inf")) - result["mae"] >= MARGIN_H)
    if (result.get("rollback") and not AUTO_PROMOTE and not (PROMOTE_MODE == "margin" and rb_clear)
            and prod_before and str(result.get("version")) != str(prod_before)):
        # fine_tune 이 예전 버전을 바로 Production 으로 올렸으면 되돌리고, 그 버전도 후보로만 둔다 (운영은 사람이 바꾼다)
        c = tr.MlflowClient()
        c.transition_model_version_stage(name=tr.MODEL_NAME, version=str(result["version"]), stage="Staging", archive_existing_versions=False)
        c.transition_model_version_stage(name=tr.MODEL_NAME, version=str(prod_before), stage="Production", archive_existing_versions=False)
        result = {**result, "promoted": False, "staged": True, "rollback_candidate": True}
    summary = {"status": "retrain_triggered", "promoted": bool(result.get("promoted")), "staged": bool(result.get("staged")),
               "rmse": result["rmse"], "mae": result["mae"], "baseline_mae": result["baseline_mae"], "prod_mae": result.get("prod_mae"),
               "rollback": bool(result.get("rollback")), "version": result.get("version"), "n_cases": len(recent), "reasons": reasons,
               "promote_mode": PROMOTE_MODE, "promote_margin_min": round(MARGIN_H * 60, 1)}
    if result.get("promoted"):
        model_loader.reset_cache()
        state.fallback = False
        how = " (rollback to earlier version)" if result.get("rollback") else ""
        _log.info(f"[OK] new_rmse={result['rmse']:.2f} - production promoted: {tr.MODEL_NAME} v{result['version']}{how}")
    elif result.get("staged"):
        _log.info(f"[INFO] candidate v{result['version']} staged (gate passed: mae={result['mae']:.2f} <= prod_mae="
                  f"{result.get('prod_mae', float('nan')):.2f}) - production kept, manual promote required")
        if reasons.get("relative"):
            rt._enter_fallback(window, "model worse than terminal plan (until candidate is promoted)")
    else:
        _log.info(f"[INFO] gate failed - keep current production (mae={result['mae']:.2f}, "
                  f"prod_mae={result.get('prod_mae', float('nan')):.2f}, baseline_mae={result['baseline_mae']:.2f})")
        if reasons.get("relative"):
            rt._enter_fallback(window, "model worse than terminal plan")
    summary["fallback"] = state.fallback
    summary["fallback_target"] = state.fallback_target if state.fallback else None
    return summary


rt._retrain_inner = _retrain_inner
