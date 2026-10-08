"""
v2.5: 바깥에 내놓는 API = 교수님 HAIC 가이드 9개(경로 이름 그대로) + 프로젝트 추가 1개. Swagger(/docs)에는 이 10개만.

  1 서빙   GET /health · POST /predict · GET /predict/next-day (LSTM + AIS 보정 + 하역시간·출항)
  2 AIOps  POST /predict/batch-test · GET /retrain/status · GET /logs · GET /logs/{filename}
  3 데이터 POST /data/upload · GET /data/status
  4 활용   GET /impact (손실·추천 + 기준표)
나머지(/v25/*, /schedule/*, /metrics/*, /admin/api/*, /predict/replay, /arrival/*, /ops/*)는 화면·관리 콘솔 내부용 — 호출은 그대로 되고 Swagger 에서만 숨김.
dashboard.py 끝에서 다른 모듈이 모두 경로를 등록한 뒤 마지막으로 import 된다.
"""
import pandas as pd
from fastapi.routing import APIRoute

from serving_app.main import app


def _find(path: str, method: str):
    for r in app.router.routes:
        if isinstance(r, APIRoute) and r.path == path and method in r.methods:
            return r  # 앞에 끼워 넣은(실제로 응답하는) 경로가 먼저 걸린다
    return None


def upcoming(as_of: str | None = None, horizon_h: float = 48.0):
    """예측 접안 = LSTM 보정, AIS가 있는 배는 위치 기반 예측과 섞은 값. 하역시간·출항 예측 포함."""
    from addons import api
    nd = api.next_day(as_of, horizon_h)
    ais = {}
    try:
        from addons import ais_eta
        a = ais_eta.ais_eta(as_of=as_of, horizon_h=horizon_h)
        ais_meta = {k: a.get(k) for k in ["ais_available", "last_fix_utc", "matched_n", "used_n", "weight_note"]}
        ais = {v.get("call"): v for v in a.get("vessels", [])}
    except Exception as e:  # AIS 모듈이 없거나 실패해도 LSTM 예측은 그대로 나간다
        ais_meta = {"ais_available": False, "error": str(e)}
    out = []
    for v in nd["vessels"]:
        x = ais.get(v["call_id"]) or {}
        w = x.get("w") or 0
        berth = x.get("ensemble") if w > 0 else v["predicted_berth"]
        dep_delay = None
        if v.get("terminal_planned_departure"):
            dep_delay = round((pd.Timestamp(v["predicted_departure"]) - pd.Timestamp(v["terminal_planned_departure"])).total_seconds() / 3600, 2)
        out.append({**v, "predicted_berth": berth, "predicted_berth_lstm": v["predicted_berth"],
                    "berth_source": "lstm+ais" if w > 0 else "lstm", "ais_weight": w, "ais_reason": x.get("reason"),
                    "ais": x.get("ais"), "departure_delay_h": dep_delay})
    return {"as_of": nd["as_of"], "horizon_h": horizon_h, "model_version": nd["model_version"], "ais": ais_meta, "vessels": out,
            "notes": ["predicted_berth: AIS 비중(ais_weight)이 0보다 크면 LSTM과 위치 기반 예측을 섞은 값 (가중치 검증 전 v0)",
                      "predicted_departure·predicted_work_h: 물량 기반 작업시간 추정 (학습 모델 아님), HJNC 평균 오차 약 3.3시간 → 참고용",
                      nd.get("note", "")]}


def impact(as_of: str | None = None, horizon_h: float = 48.0, terminal: str = "ALL", delay_h: float = 2.0):
    from addons import impact, loss
    return {**impact.impact(as_of=as_of, horizon_h=horizon_h, terminal=terminal),
            "loss_table": loss.loss_table(delay_h)}


def status():
    from addons import api
    from serving_app.routers import health
    b = None
    try:
        from addons import dashboard as dv
        days = dv.board()["days"]
        b = {"date": days[-1]["iso"], "all": days[-1]["win"], "by_terminal": {t: x["win"] for t, x in days[-1]["by_term"].items()}} if days else None
    except Exception:
        pass
    return {"health": health.health(), "retrain": api.retrain_status(), "rules": api.gates(), "latest_window": b}


from serving_app.routers.data import upload as _upload
from serving_app.routers.predict import batch_test as _batch_test
from serving_app.routers.predict import predict as _predict

from serving_app.routers import health as _health
from serving_app.routers import logs as _logs
from serving_app.routers.data import status as _data_status

# 바깥에 내놓는 API = 교수님 HAIC 가이드의 7개(이름 그대로) + 이 프로젝트 핵심 3개.  (경로, 메서드, 그룹, 설명, 함수, Swagger 공개)
ROUTES = [
    ("/health", "GET", "1. 서빙 (교수님 구조)", "헬스체크: 모델 로딩 여부·Lazy/Eager·폴백·재학습 중 여부", _health.health, True),
    ("/predict", "POST", "1. 서빙 (교수님 구조)", "배 1척 접안 지연 예측 (같은 항로 최근 20척 → LSTM 보정)", _predict, True),
    ("/predict/next-day", "GET", "1. 서빙 (교수님 구조)", "다음 48시간 접안 예정 배: 예측 접안(LSTM + AIS 보정) + 하역시간·출항. as_of 과거면 실제도", upcoming, True),
    ("/predict/batch-test", "POST", "2. AIOps (교수님 구조)", "정답 확정 배 → 예측 → 최근 14일 드리프트 판정 → 비동기 재학습", _batch_test, True),
    ("/retrain/status", "GET", "2. AIOps (교수님 구조)", "비동기 재학습 결과, Production 버전·승격 시각", ("/retrain/status", "GET"), True),
    ("/logs", "GET", "2. AIOps (교수님 구조)", "로그 파일 목록 (aiops.log, requests.log)", _logs.list_logs, True),
    ("/logs/{filename}", "GET", "2. AIOps (교수님 구조)", "로그 내용: [WARN] drift detected → [INFO] retrain triggered → [OK] promoted", _logs.read_log, True),
    ("/data/upload", "POST", "3. 데이터 (교수님 구조)", "선석 스케줄 업로드 (HJNC 엑셀 또는 터미널 통합 CSV)", _upload, True),
    ("/data/status", "GET", "3. 데이터 (교수님 구조)", "업로드 파일 현황 (척수, 정답 확정 배, 터미널 예정 오차)", _data_status, True),
    ("/impact", "GET", "4. 활용 (프로젝트 추가)", "배별 예상 손실·추천 행동 + 지연 시간별 손실 기준표", impact, True),
    # 같은 기능의 다른 이름 (호환용, Swagger 비공개)
    ("/arrival/predict", "POST", "", "", _predict, False),
    ("/arrival/upcoming", "GET", "", "", upcoming, False),
    ("/ops/batch-test", "POST", "", "", _batch_test, False),
    ("/ops/status", "GET", "", "", status, False),
    ("/ops/history", "GET", "", "", ("/metrics/history", "GET"), False),
]

public = []
for path, method, tag, summary, target, show in ROUTES:
    fn = target if callable(target) else _find(*target).endpoint
    r = APIRoute(path, fn, methods=[method], tags=[tag] if tag else None, summary=summary or None)
    app.router.routes.insert(0, r)  # 같은 경로는 앞에 끼운 것이 응답 (기존 동작과 같거나 칸만 늘어남)
    if show:
        public.append(r)


def _openapi():
    """Swagger(/docs)에는 바깥용 7개만. 나머지는 화면·관리 콘솔 내부용이라 숨긴다 (호출은 그대로 된다)."""
    if app.openapi_schema is None:
        from fastapi.openapi.utils import get_openapi
        app.openapi_schema = get_openapi(title="부산신항 접안 지연 조기경보 API", version="2.5",
                                         description="교수님 HAIC 구조 9개 + 프로젝트 추가 1개. 명세: docs/API_명세서_v2.5.md", routes=public)
    return app.openapi_schema


app.openapi_schema = None
app.openapi = _openapi
