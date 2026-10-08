"""
드리프트 → 자동 재학습 → (실패 시) 터미널 예정으로 되돌리기(폴백).

  감지 → [WARN] → 최근 21일 동안 접안이 끝나 정답이 확정된 배로 fine-tuning(빈칸 10: warm start)
       → 게이트(현재 Production보다 나쁘지 않음 + MAE ≤ 1.0h) 통과: Production 승격 + [OK] (빈칸 11: promoted일 때만)
       → 미통과 + 모델이 터미널 예정보다 못함: 폴백 (예측 대신 터미널 예정을 그대로 응답)
  드리프트가 사라지면 폴백 해제.

예제와 다른 점
  - 빈칸 9 '최근 21+SEQ_LEN행' → '최근 21일 접안 확정 배'. 이 모델은 샘플마다 이미 20척 이력을 들고 있어
    별도 창 길이를 더할 필요가 없다.
  - 시뮬레이션(inject_shift_h)일 때는 재학습 창에도 같은 지연을 더한다. 예제는 합성 데이터로 감지만 하고
    재학습은 실제 데이터로 해서 '바뀐 세상'을 배우지 않는다. 여기서는 바뀐 세상을 21일 겪었다고 가정한다.
  - 폴백 추가: 합성 실험에서 재학습이 실패한 뒤 낡은 모델을 계속 쓰면 터미널 예정보다 나빠졌다.
로그 문장 [WARN]/[INFO]/[OK] 3개는 대시보드가 읽으니 글자를 바꾸지 않는다.

v1.1: 재학습을 요청 밖으로 뺐다 (교수님 Wrap-Up 1번 "요청 안에서 동기로 재학습").
  RETRAIN_MODE=async(기본): 감지하면 별도 스레드에서 재학습, 응답은 바로 "retrain_scheduled".
                           relative 드리프트(모델이 터미널 예정보다 못함)면 재학습이 끝날 때까지 바로 폴백.
                           진행·결과는 GET /retrain/status
  RETRAIN_MODE=sync       : 예제처럼 요청 안에서 끝까지 기다림
"""
import logging
import os
import threading
from datetime import timedelta

import pandas as pd

from serving_app import state
from serving_app.monitoring.drift_detector import drift_reasons, is_drift

logger = logging.getLogger("aiops")
RETRAIN_DAYS = int(os.getenv("RETRAIN_DAYS", "21"))


_lock = threading.Lock()


def _choose_fallback(recent_predictions: list[dict]) -> str:
    """v2.2 폴백 2단: 최근 창에서 단순 평균(mean20)이 터미널 예정보다 나으면 mean20, 아니면 터미널 예정."""
    from serving_app.monitoring.drift_detector import compute_mae
    w = [p for p in recent_predictions if "mean20" in p]
    if w and compute_mae(w, "mean20") < compute_mae(w, "baseline"):
        return "mean20"
    return "terminal"


def _enter_fallback(recent_predictions: list[dict], why: str):
    state.fallback = True
    state.fallback_target = _choose_fallback(recent_predictions)
    target = "route mean20" if state.fallback_target == "mean20" else "terminal plan"
    logger.warning(f"[WARN] fallback to {target} - {why}")


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    reasons = drift_reasons(recent_predictions)
    if not is_drift(recent_predictions):
        if state.fallback:
            state.fallback = False
            logger.info("[INFO] fallback cleared - model beats terminal plan again")
        return {"status": "ok", "fallback": state.fallback}

    logger.warning("[WARN] drift detected - triggering retrain")

    if state.retrain_running:
        logger.info("[INFO] retrain already running - skip")
        return {"status": "retrain_in_progress", "reasons": reasons, "fallback": state.fallback}
    logger.info(f"[INFO] retrain triggered (window=last_{RETRAIN_DAYS}_days, reasons={reasons})")

    times = [pd.Timestamp(p["decision_time"]) for p in recent_predictions if p.get("decision_time")]
    if os.getenv("RETRAIN_MODE", "async") == "sync":
        return _retrain(times, reasons, state.sim_shift_h, list(recent_predictions))
    if reasons["relative"] and not state.fallback:
        _enter_fallback(recent_predictions, "model worse than terminal plan (until retrain finishes)")
    state.retrain_running = True
    window = list(recent_predictions)
    threading.Thread(target=_retrain, args=(times, reasons, state.sim_shift_h, window), daemon=True).start()
    return {"status": "retrain_scheduled", "reasons": reasons, "fallback": state.fallback}


def _retrain(times: list, reasons: dict, sim_shift_h: float = 0.0, window: list | None = None) -> dict:
    with _lock:
        state.retrain_running = True
        try:
            summary = _retrain_inner(times, reasons, sim_shift_h, window or [])
        except Exception as e:  # 재학습 실패가 서빙을 죽이지 않게
            logger.exception(f"[INFO] retrain failed - keep current production ({e})")
            summary = {"status": "retrain_failed", "error": str(e), "reasons": reasons, "fallback": state.fallback}
        finally:
            state.retrain_running = False
        state.last_retrain = {**summary, "finished_at": pd.Timestamp.now().isoformat(timespec="seconds")}
        return summary


def _retrain_inner(times: list, reasons: dict, sim_shift_h: float, window: list) -> dict:
    from data.features import load_samples
    from data.storage import latest_upload
    from serving_app.train_and_register import fine_tune

    samples = load_samples(latest_upload())
    now = max(times) if times else samples[-1]["decision_time"]
    recent = [s for s in samples if now - timedelta(days=RETRAIN_DAYS) <= s["decision_time"] < now
              and s["actual_berth"] <= now]
    if sim_shift_h:
        import copy
        recent = copy.deepcopy(recent)
        for s in recent:
            s["actual_correction"] += sim_shift_h
            for pt in s["sequence"]:
                pt["delay_hours"] += sim_shift_h
        logger.info(f"[INFO] simulation: +{sim_shift_h}h shift applied to retrain window")
    if len(recent) < 10:
        logger.info(f"[INFO] too few labeled calls for retrain ({len(recent)}) - keep current production")
        return {"status": "retrain_skipped", "n_cases": len(recent), "reasons": reasons, "fallback": state.fallback}

    result = fine_tune(recent)
    summary = {"status": "retrain_triggered", "promoted": bool(result.get("promoted")), "rmse": result["rmse"],
               "mae": result["mae"], "baseline_mae": result["baseline_mae"], "prod_mae": result.get("prod_mae"), "rollback": bool(result.get("rollback")), "version": result.get("version"), "n_cases": len(recent),
               "reasons": reasons}
    if result.get("promoted"):
        from serving_app import model_loader
        model_loader.reset_cache()
        state.fallback = False
        how = " (rollback to earlier version)" if result.get("rollback") else ""
        logger.info(f"[OK] new_rmse={result['rmse']:.2f} - production promoted: BERTH_ARRIVAL_Corrector v{result['version']}{how}")
    else:
        logger.info(f"[INFO] gate failed - keep current production (mae={result['mae']:.2f}, "
                    f"prod_mae={result.get('prod_mae', float('nan')):.2f}, baseline_mae={result['baseline_mae']:.2f})")
        if reasons["relative"]:
            _enter_fallback(window, "model worse than terminal plan")
    summary["fallback"] = state.fallback
    summary["fallback_target"] = state.fallback_target if state.fallback else None
    return summary
