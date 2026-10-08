"""
v2.5 관리자 콘솔 (/admin). AIOps 운영·정확도 검증·데이터 업로드를 사용자 화면(/early-warning)에서 분리.
화면 구성은 교수님 AIOps 데모 콘솔(Dashboard / Simulation / Datasets / System)을 따른다. 기존 serving_app 파일은 고치지 않는다.

  GET  /admin                              관리자 콘솔 화면 (addons/admin.html)
  GET  /admin/api/summary?window=5m|1h|6h|24h  요청 수·평균 응답시간·성공률(요청 로그 집계) + 모델 오차 추이 + 드리프트 상태
  GET  /admin/api/models                   MLflow 등록 버전 이력 (버전, 등록 시각, 학습 방식, MAE, 스테이지)
  POST /admin/api/models/{version}/promote  지정 버전을 Production으로 (시연 뒤 v1 되돌리기용)
  GET  /admin/api/alerts?n=30              logs/aiops.log 최근 이벤트
  POST /admin/api/datasets/refresh         addons/refresh.py 실행 (HJNC·PNIT·HPNT 다시 받기 → 통합 CSV → 업로드)
  GET  /admin/api/datasets/refresh         마지막 갱신 결과
  GET  /admin/api/system                   서버 상태·환경변수·버전·데이터 파일

요청 로그: 이 모듈이 미들웨어로 업무 요청(접안 예측·일정표·손실 계산 API)만 logs/requests.log 에 한 줄 JSON으로 남긴다.
  관리자 콘솔이 부르는 호출(X-Client: admin-console)과 헬스체크·로그·재학습 상태·드리프트 시연은 세지 않는다.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque

import pandas as pd
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute

from serving_app import model_loader, state
from serving_app.main import app

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "admin.html")
REQ_LOG = "logs/requests.log"
AIOPS_LOG = "logs/aiops.log"
STARTED = time.time()
WINDOWS = {"5m": 300, "1h": 3600, "6h": 6 * 3600, "24h": 86400}
_reqs: deque = deque(maxlen=200_000)
_refresh: dict = {"running": False, "last": None}
# 운영 지표는 업무 요청만 센다: 접안 예측·일정표·손실 계산을 실제로 쓰는 호출.
# 화면 파일, 헬스체크·로그·재학습 상태 같은 운영 호출, 드리프트 시연(/predict/replay, batch-test), 관리자 콘솔이 보낸 호출(X-Client 헤더)은 뺀다
_BUSINESS = re.compile(r"^/(predict|predict/next-day|v25/board|v25/impact|v25/loss-table|v25/ais-eta|v25/upstream-queue|schedule/[\w-]+|metrics/upstream)$")

os.makedirs("logs", exist_ok=True)
if os.path.exists(REQ_LOG):  # 재시작해도 24시간 안 기록은 이어서 보이게
    cut = time.time() - WINDOWS["24h"]
    for ln in open(REQ_LOG, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(ln)
            if r["ts"] >= cut and _BUSINESS.match(r["path"]):
                _reqs.append(r)
        except Exception:
            pass


@app.middleware("http")
async def _record(request: Request, call_next):
    t = time.perf_counter()
    status = 500
    try:
        resp = await call_next(request)
        status = resp.status_code
        return resp
    finally:
        path = request.url.path
        if _BUSINESS.match(path) and request.headers.get("x-client") != "admin-console":
            r = {"ts": time.time(), "method": request.method, "path": path, "status": status,
                 "ms": round((time.perf_counter() - t) * 1000, 1)}
            _reqs.append(r)
            with open(REQ_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")


def _series(rows, span, n=12):
    """구간을 n칸으로 나눠 칸마다 요청 수·평균 응답시간·성공률."""
    now, step = time.time(), span / n
    out = []
    for i in range(n):
        lo, hi = now - span + i * step, now - span + (i + 1) * step
        b = [r for r in rows if lo <= r["ts"] < hi]
        out.append({"count": len(b), "ms": round(sum(r["ms"] for r in b) / len(b), 1) if b else 0,
                    "ok": round(sum(r["status"] < 500 for r in b) / len(b), 4) if b else None})
    return out


def summary(window: str = "5m"):
    from addons import dashboard as dv
    span = WINDOWS.get(window, 300)
    rows = [r for r in _reqs if r["ts"] >= time.time() - span]
    paths = {}
    for r in rows:
        paths[r["path"]] = paths.get(r["path"], 0) + 1
    b = dv.board()
    days = b["days"][-30:]
    last = days[-1]["win"] if days else {}
    retrain = state.last_retrain
    return {"window": window,
            "requests": len(rows), "avg_ms": round(sum(r["ms"] for r in rows) / len(rows), 1) if rows else 0,
            "success_rate": round(sum(r["status"] < 500 for r in rows) / len(rows), 4) if rows else None,
            "series": _series(rows, span), "top_paths": sorted(paths.items(), key=lambda x: -x[1])[:6],
            "model": {"version": b["model_version"], "mae_21": last.get("mae"), "terminal_mae_21": last.get("term_mae"),
                      "rmse_21": last.get("rmse"), "drift_condition": last.get("drift"),
                      "trend": [{"date": d["iso"], "mae_21": d["win"].get("mae"), "terminal_mae_21": d["win"].get("term_mae")} for d in days]},
            "drift": {"retrain_running": state.retrain_running, "fallback": state.fallback,
                      "last": retrain}}


def _client():
    from mlflow.tracking import MlflowClient
    return MlflowClient()


def models():
    from serving_app.train_and_register import MODEL_NAME
    try:
        c = _client()
        vs = c.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        return {"name": MODEL_NAME, "available": False, "error": str(e), "versions": []}
    out = []
    for v in sorted(vs, key=lambda v: -int(v.version)):
        m, p = {}, {}
        try:
            run = c.get_run(v.run_id)
            m, p = run.data.metrics, run.data.params
        except Exception:
            pass
        out.append({"version": int(v.version), "stage": v.current_stage,
                    "created": pd.Timestamp(v.creation_timestamp, unit="ms", tz="UTC").tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M"),
                    "updated": pd.Timestamp(v.last_updated_timestamp or v.creation_timestamp, unit="ms", tz="UTC").tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M"),
                    "mode": p.get("mode"), "n_cases": p.get("n_cases") or p.get("n_train"),
                    "mae": m.get("mae"), "rmse": m.get("rmse"), "baseline_mae": m.get("baseline_mae"), "prod_mae": m.get("prod_mae")})
    return {"name": MODEL_NAME, "available": True, "versions": out,
            "serving": getattr(model_loader._model_cache, "version", None)}


def promote(version: int):
    from serving_app.train_and_register import MODEL_NAME
    if os.getenv("MODEL_SOURCE", "local") != "mlflow":
        raise HTTPException(400, "MODEL_SOURCE=mlflow 일 때만 버전을 바꿀 수 있습니다")
    c = _client()
    if not any(int(v.version) == version for v in c.search_model_versions(f"name='{MODEL_NAME}'")):
        raise HTTPException(404, f"v{version} 없음")
    c.transition_model_version_stage(name=MODEL_NAME, version=str(version), stage="Production", archive_existing_versions=True)
    model_loader.reset_cache()
    state.fallback = False
    import logging
    logging.getLogger("aiops").info(f"[INFO] manual promote by admin console - production: {MODEL_NAME} v{version}")
    return {"promoted": version}


def alerts(n: int = 30):
    if not os.path.exists(AIOPS_LOG):
        return {"alerts": []}
    lines = [ln.rstrip("\n") for ln in open(AIOPS_LOG, encoding="utf-8", errors="ignore") if ln.strip()]
    out = []
    for ln in lines[-n:][::-1]:
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?\d* \[(\w+)\] (.*)", ln)
        if m:
            msg = m.group(3)
            lv = "OK" if ("[OK]" in msg or "staged" in msg or "manual promote" in msg) else "WARN" if "[WARN]" in msg else "FAIL" if "fail" in msg.lower() else m.group(2)
            out.append({"time": m.group(1), "level": lv, "message": msg})
        else:
            out.append({"time": None, "level": "INFO", "message": ln})
    return {"alerts": out}


def _run_refresh():
    _refresh["running"] = True
    t = time.time()
    try:
        p = subprocess.run([sys.executable, "addons/refresh.py", "--server", f"http://127.0.0.1:{os.getenv('PORT', '8099')}"],
                           capture_output=True, text=True, timeout=600)
        out = "\n".join(ln for ln in (p.stdout + p.stderr).splitlines() if "default style" not in ln)
        _refresh["last"] = {"ok": p.returncode == 0, "output": out[-3000:], "seconds": round(time.time() - t, 1),
                            "finished_at": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        _refresh["last"] = {"ok": False, "output": str(e), "finished_at": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")}
    finally:
        _refresh["running"] = False


def refresh_start():
    if _refresh["running"]:
        return {"status": "running"}
    threading.Thread(target=_run_refresh, daemon=True).start()
    return {"status": "started"}


def refresh_status():
    return _refresh


def system():
    import platform

    from addons.rules import RULES
    from data.storage import latest_upload
    from addons import drift as dr
    from serving_app.monitoring.drift_detector import ABS_RMSE_THRESHOLD, REL_TOLERANCE_HOURS
    versions = {"python": platform.python_version()}
    for mod in ("fastapi", "tensorflow", "mlflow", "pandas"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        f = latest_upload()
        data = {"file": os.path.basename(f), "size_kb": round(os.path.getsize(f) / 1024, 1),
                "uploaded_at": pd.Timestamp(os.path.getmtime(f), unit="s", tz="UTC").tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")}
    except FileNotFoundError:
        data = None
    env = {k: os.getenv(k) for k in ("MODEL_SOURCE", "LOADING_MODE", "DRIFT_MODE", "RETRAIN_MODE", "RETRAIN_DAYS", "MODEL_POLL_SEC", "TZ")}
    return {"status": "ok", "model_loaded": model_loader._model_cache is not None,
            "serving_version": getattr(model_loader._model_cache, "version", None),
            "fallback": state.fallback, "fallback_target": state.fallback_target if state.fallback else None,
            "retrain_running": state.retrain_running, "uptime_s": int(time.time() - STARTED), "env": env, "versions": versions,
            "data": data, "logs": sorted(os.listdir("logs")) if os.path.isdir("logs") else [],
            "drift_rule": {"window_days": dr.WINDOW_DAYS, "min_ships": dr.MIN_SHIPS, "bias_h": dr.BIAS_H, "retrain_days": dr.RETRAIN_DAYS,
                           "rel_tolerance_h": REL_TOLERANCE_HOURS, "abs_rmse_h": ABS_RMSE_THRESHOLD},
            "rules": {k: {"value": str(v["value"]), "confirmed": v["confirmed"]} for k, v in RULES.items()}}


def page():
    return FileResponse(HTML, media_type="text/html")


for _path, _fn, _m in (("/admin", page, "GET"), ("/admin/api/summary", summary, "GET"), ("/admin/api/models", models, "GET"),
                       ("/admin/api/models/{version}/promote", promote, "POST"), ("/admin/api/alerts", alerts, "GET"),
                       ("/admin/api/datasets/refresh", refresh_start, "POST"), ("/admin/api/datasets/refresh", refresh_status, "GET"),
                       ("/admin/api/system", system, "GET")):
    app.router.routes.insert(0, APIRoute(_path, _fn, methods=[_m]))
