"""
v2.5: 팀 화면 연결용 API (팀원 요청 1~7). 기존 serving_app 파일은 고치지 않고 경로를 앞에 끼워 넣는다.

  1 GET /metrics/history?days=60&terminal=ALL   일별 운영 기록 (실제·예측 평균 지연, 최근 21척 MAE, 드리프트·승격)
  2 GET /metrics/gates                         드리프트 판정 기준, 배포 게이트 기준
  3 GET /retrain/status                        기존 응답 + production_version, promoted_at (MLflow)
  4 GET /predict/next-day?as_of=&horizon_h=     기존 응답 칸 + 터미널, 양하·선적·S/H, 반입 마감, 선종·국적·총톤수, 전출항지·차항지, 예선
  5 GET /metrics/upstream?days=60               앞 항만 컨테이너선 입항 척수 (IMF PortWatch, 대기 척수 아님, 약 6일 늦음)
  6 GET /schedule/reschedule?as_of=&horizon_h=  일정표 조율 (규칙 기반 v0: 같은 선석 앞 배 예측 출항 + 최소 간격 2시간)
  7 GET /schedule/berth-allocation?as_of=...     선석 점유·겹침·추천 선석 (규칙 기반 v0)

6·7은 학습한 모델이 아니라 업무 규칙(addons/rules.py min_gap_between_vessels_h) 계산이다. 응답에 method로 적어 둔다.
HJNC 엑셀의 선석 번호(1B~4B)는 물리 선석이 아니라 안벽 위치 구분이라 겹침은 "확인 신호"로만 쓴다 (인계문서 10절).
"""
import os
import re

import pandas as pd
from fastapi.routing import APIRoute

from addons import dashboard as dv
from addons.rules import RULES
from data.storage import latest_upload
from serving_app import model_loader, state
from serving_app.main import app
from serving_app.routers import retrain as retrain_router
from serving_app.schedule import WorkTimeEstimator
from serving_app.train_and_register import ABS_MAE_GATE, MODEL_NAME

PORTWATCH = "data/real/imf_portwatch_daily_portcalls.csv"
LOG = "logs/aiops.log"
GAP_H = RULES["min_gap_between_vessels_h"]["value"]


def _now():
    return pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).floor("min")


def _log_events() -> dict:
    """aiops.log → 날짜별 {drift, promoted, versions}. 로그 시각은 서버가 실제로 겪은 시각."""
    ev: dict = {}
    if not os.path.exists(LOG):
        return ev
    for ln in open(LOG, encoding="utf-8", errors="ignore"):
        d = ln[:10]
        if not re.match(r"\d{4}-\d{2}-\d{2}", d):
            continue
        e = ev.setdefault(d, {"drift": False, "promoted": False, "versions": []})
        if "[WARN] drift detected" in ln:
            e["drift"] = True
        m = re.search(r"production promoted: \S+ v(\d+)", ln)
        if m:
            e["promoted"] = True
            e["versions"].append(int(m.group(1)))
    return ev


def metrics_history(days: int = 60, terminal: str = "ALL"):
    b = dv.board()
    ev = _log_events()
    out = []
    for d in b["days"][-days:]:
        s = d if terminal == "ALL" else d["by_term"].get(terminal)
        if not s:
            continue
        w = s.get("win", {})
        e = ev.get(d["iso"], {})
        out.append({"date": d["iso"], "n_ships": s.get("n_ships", 0),
                    "actual_mean_delay_h": s.get("actual"), "predicted_mean_delay_h": s.get("pred"),
                    "mae_21": w.get("mae"), "terminal_mae_21": w.get("term_mae"), "rmse_21": w.get("rmse"),
                    "drift_condition": bool(w.get("drift")), "drift_detected": bool(w.get("drift") or e.get("drift")),
                    "promoted": bool(e.get("promoted")), "promoted_versions": e.get("versions", []),
                    "model_version": b["model_version"], "calls_by_terminal": d.get("calls_by_term", {})})
    return {"terminal": terminal, "days": out,
            "note": "predicted_mean_delay_h = 그날 접안한 배마다 반입 마감(접안 10시간 전) 시점에 낸 예측의 평균. "
                    "과거 일자도 지금 서빙 중인 모델로 그 시점 입력만 넣어 다시 계산한 값이라 model_version은 현재 버전. "
                    "promoted·drift_detected의 로그 근거는 서버가 실제로 겪은 날짜(logs/aiops.log)",
            "terminals": list(b["terminals"])}


def gates():
    b = dv.board()
    last = b["days"][-1]["win"] if b["days"] else {}
    return {"drift": {"window": dv.WINDOW_SIZE, "window_days": dv.dr.WINDOW_DAYS, "min_ships": dv.dr.MIN_SHIPS, "bias_h": round(dv.dr.BIAS_H, 3),
                      "retrain_days": dv.dr.RETRAIN_DAYS, "rel_tolerance_h": dv.REL_TOLERANCE_HOURS, "abs_rmse_h": dv.ABS_RMSE_THRESHOLD,
                      "rule": "터미널별 최근 14일(15척 이상): MAE ≥ 터미널 예정 MAE + 3분, 또는 RMSE > 60분, 또는 |평균(실제 − 모델)| ≥ 20분",
                      "current_line_h": round((last.get("term_mae") or 0) + dv.REL_TOLERANCE_HOURS, 3)},
            "deploy_gate": {"abs_mae_h": ABS_MAE_GATE,
                            "initial": "새 모델 MAE < 터미널 예정 MAE 이고 MAE <= abs_mae_h",
                            "fine_tune": "새 모델 MAE <= 운영 모델 MAE 이고 MAE <= abs_mae_h (아니면 예전 버전 되돌리기 또는 폴백)"},
            "ship_grades_h": list(dv.GRADES), "units": "hours"}


def retrain_status():
    base = retrain_router.status()
    prod, at = None, None
    if os.getenv("MODEL_SOURCE", "local") == "mlflow":
        try:
            from mlflow.tracking import MlflowClient
            vs = [v for v in MlflowClient().search_model_versions(f"name='{MODEL_NAME}'") if v.current_stage == "Production"]
            if vs:
                v = max(vs, key=lambda v: int(v.version))
                prod = int(v.version)
                ts = v.last_updated_timestamp or v.creation_timestamp
                at = pd.Timestamp(ts, unit="ms", tz="UTC").tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    return {**base, "production_version": prod, "promoted_at": at}


def _estimators() -> dict:
    key = dv._cache.get("split_key")
    if dv._cache.get("est_key") != key:
        dv._cache["est"] = {t: WorkTimeEstimator(df) for t, (df, _) in dv._split(latest_upload()).items() if len(df)}
        dv._cache["est_key"] = key
    return dv._cache["est"]


def _rows(as_of: str | None, horizon_h: float):
    model = model_loader.get_model()
    parts = dv._split(latest_upload())
    t0 = pd.Timestamp(as_of) if as_of else _now()
    up, rec, _ = dv._window(parts, t0, horizon_h, model)
    return parts, t0, up, rec, model


def next_day(as_of: str | None = None, horizon_h: float = 48.0):
    parts, t0, up, _, model = _rows(as_of, horizon_h)
    est = _estimators()
    info = dv._vessel_info()
    out = []
    for s in up:
        df = parts[s["terminal"]][0]
        row = df[df["모선항차"] == s["call"]].iloc[0]
        e = est[s["terminal"]]
        berth = pd.Timestamp(s["predicted"])
        dep = e.departure(berth, s["box"], s["route"], t0)
        vi = info.get(s["name"].upper(), {})
        cct = row.get("반입 마감(원본)", row["반입 마감일시"]) if "반입 마감(원본)" in row else row["반입 마감일시"]
        out.append({"call_id": s["call"], "terminal": s["terminal"], "vessel": s["name"], "berth_no": s["berth"],
                    "line": s["line"], "route": s["route"], "box": s["box"],
                    "terminal_planned_berth": s["planned"], "predicted_berth": s["predicted"], "predicted_delay_h": s["delay"],
                    "predicted_berth_80": [(berth + pd.Timedelta(hours=-0.31)).strftime("%H:%M"), (berth + pd.Timedelta(hours=0.33)).strftime("%H:%M")],
                    "terminal_planned_departure": row["출항일시"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["출항일시"]) else None,
                    "predicted_departure": dep.strftime("%Y-%m-%d %H:%M"),
                    "predicted_work_h": round(e.work_hours(s["box"], s["route"], t0), 1),
                    "actual_berth": s["actual_berth"], "actual_delay_h": s["actual"],
                    "unload": s["unload"], "load": s["load"], "shift": s["shift"],
                    "cargo_cutoff": None if pd.isna(cct) else pd.Timestamp(cct).strftime("%Y-%m-%d %H:%M"),
                    "vessel_type": vi.get("type"), "flag": vi.get("flag"), "gross_tonnage": vi.get("gt"),
                    "prev_port": vi.get("prev_port_last"), "next_port": vi.get("next_port_last"),
                    "vessel_info_seen": vi.get("last_seen"), "tugs": s["tugs"], "compulsory_pilot": s["pilot"]})
    return {"as_of": t0.strftime("%Y-%m-%d %H:%M"), "model_version": ("fallback:" + state.fallback_target) if state.fallback else model.version,
            "vessels": out,
            "note": "선종·국적·총톤수·전출항지·차항지는 부산항만공사 선박 입출항 집계(2019~2024)를 배 이름으로 찾은 값 (없으면 null, "
                    "같은 이름의 다른 배일 수 있음). 80% 범위는 HJNC 기준(residual_band). as_of가 과거면 actual_* 가 채워진다"}


def upstream(days: int = 60):
    if not os.path.exists(PORTWATCH):
        return {"available": False}
    d = pd.read_csv(PORTWATCH, parse_dates=["date"])
    piv = d.pivot_table(index="date", columns="portname", values="portcalls_container", aggfunc="sum").sort_index()
    ma = piv.rolling(7, min_periods=1).mean()
    base = piv[piv.index.year == 2025].mean()
    rows = []
    for t in piv.index[-days:]:
        rows.append({"date": t.strftime("%Y-%m-%d"),
                     **{f"{p.lower()}_container_calls": int(piv.at[t, p]) for p in piv.columns},
                     **{f"{p.lower()}_ma7": round(float(ma.at[t, p]), 1) for p in piv.columns},
                     **{f"{p.lower()}_vs_2025_pct": round(float(ma.at[t, p] / base[p] * 100), 1) for p in piv.columns}})
    return {"available": True, "metric": "컨테이너선 하루 입항 척수 (대기 척수 아님)", "source": "IMF PortWatch (AIS 기반)",
            "last_date": piv.index.max().strftime("%Y-%m-%d"), "lag_note": "공개 지연 약 6일", "ports": list(piv.columns),
            "days": rows}


def _occupancy(parts, t0, up, rec):
    """선석 점유 구간: 접안 예정 배(예측 접안 → 예측 출항) + 지금 접안해 있는 배(실제 접안 → 터미널 예정 출항)."""
    est = _estimators()
    occ = []
    for s in up:
        e = est[s["terminal"]]
        b = pd.Timestamp(s["predicted"])
        occ.append({"call_id": s["call"], "terminal": s["terminal"], "vessel": s["name"], "berth_no": s["berth"],
                    "start": b, "end": e.departure(b, s["box"], s["route"], t0), "state": "예정", "delay_h": s["delay"]})
    for term, (df, _) in parts.items():
        w = df[df["작업 시작일시"].notna() & (df["입항일시"] <= t0) & (df["출항일시"] > t0)]
        for _, r in w.iterrows():
            occ.append({"call_id": r["모선항차"], "terminal": term, "vessel": r["선박명"], "berth_no": str(r["선석"]),
                        "start": r["입항일시"], "end": r["출항일시"], "state": "접안 중", "delay_h": 0.0})
    return occ


def _berth_key(b: str) -> str:
    return re.sub(r"\(.*\)", "", str(b)).strip()


def reschedule(as_of: str | None = None, horizon_h: float = 48.0):
    parts, t0, up, rec, _ = _rows(as_of, horizon_h)
    occ = _occupancy(parts, t0, up, rec)
    out = []
    for (term, bk), g in pd.DataFrame(occ).assign(bk=lambda x: x["berth_no"].map(_berth_key)).groupby(["terminal", "bk"]):
        prev_end, prev_name = None, None
        for _, o in g.sort_values("start").iterrows():
            start, end = o["start"], o["end"]
            reason = []
            if o["state"] == "예정" and abs(o["delay_h"]) >= 0.25:
                reason.append(f"예측 지연 {o['delay_h'] * 60:+.0f}분")
            if o["state"] == "예정" and prev_end is not None and start < prev_end + pd.Timedelta(hours=GAP_H):
                push = prev_end + pd.Timedelta(hours=GAP_H) - start
                start, end = start + push, end + push
                reason.append(f"같은 선석 앞 배 {prev_name} 출항 예측 {prev_end:%m-%d %H:%M} + {GAP_H:g}시간 간격")
            if o["state"] == "예정":
                out.append({"call_id": o["call_id"], "terminal": term, "vessel": o["vessel"],
                            "rescheduled_berth": start.strftime("%Y-%m-%d %H:%M"), "rescheduled_departure": end.strftime("%Y-%m-%d %H:%M"),
                            "rescheduled_berth_no": o["berth_no"], "change_reason": " / ".join(reason) or "변경 없음"})
            prev_end, prev_name = end, o["vessel"]
    return {"as_of": t0.strftime("%Y-%m-%d %H:%M"), "method": f"규칙 기반 v0 (학습 모델 아님): 예측 접안 순서대로, 같은 선석 앞 배 예측 출항 + {GAP_H:g}시간 뒤로 민다",
            "berth_note": "HJNC 선석 번호(1B~4B)는 물리 선석이 아니라 안벽 위치 구분 → 겹침은 확인 신호로만",
            "vessels": sorted(out, key=lambda x: x["rescheduled_berth"])}


def berth_allocation(as_of: str | None = None, horizon_h: float = 48.0):
    parts, t0, up, rec, _ = _rows(as_of, horizon_h)
    occ = _occupancy(parts, t0, up, rec)
    gap = pd.Timedelta(hours=GAP_H)
    berths = {t: sorted({_berth_key(b) for b in df["선석"].dropna().astype(str) if b and b != "nan"}) for t, (df, _) in parts.items()}
    out = []
    for o in occ:
        if o["state"] != "예정":
            continue
        same = [p for p in occ if p is not o and p["terminal"] == o["terminal"]]
        def clashes(bk):
            return [p for p in same if _berth_key(p["berth_no"]) == bk and p["start"] < o["end"] + gap and o["start"] < p["end"] + gap]
        own = clashes(_berth_key(o["berth_no"]))
        rec_b = _berth_key(o["berth_no"])
        if own:
            free = [b for b in berths.get(o["terminal"], []) if b != rec_b and not clashes(b)]
            if free:
                rec_b = free[0]
        out.append({"call_id": o["call_id"], "terminal": o["terminal"], "vessel": o["vessel"], "berth_no": o["berth_no"],
                    "recommended_berth_no": rec_b, "occupancy_start": o["start"].strftime("%Y-%m-%d %H:%M"),
                    "occupancy_end": o["end"].strftime("%Y-%m-%d %H:%M"),
                    "conflict_with": [{"call_id": p["call_id"], "vessel": p["vessel"], "state": p["state"],
                                       "overlap_h": round(max(0.0, (min(o["end"], p["end"]) - max(o["start"], p["start"])).total_seconds() / 3600), 2),
                                       "gap_short_h": round(max(0.0, (gap - abs((o["start"] - p["end"]) if o["start"] >= p["end"] else (p["start"] - o["end"]))).total_seconds() / 3600), 2)
                                       if not (o["start"] < p["end"] and p["start"] < o["end"]) else None}
                                      for p in own]})
    return {"as_of": t0.strftime("%Y-%m-%d %H:%M"),
            "method": f"규칙 기반 v0 (학습 모델 아님): 예측 접안~예측 출항 구간이 같은 선석의 다른 배와 겹치거나 간격이 {GAP_H:g}시간 미만이면 conflict, 비어 있는 다른 선석을 추천",
            "berths": berths, "berth_note": "HJNC 선석 번호(1B~4B)는 물리 선석이 아니라 안벽 위치 구분 → 겹침은 확인 신호로만",
            "vessels": sorted(out, key=lambda x: x["occupancy_start"])}


for _path, _fn in (("/metrics/history", metrics_history), ("/metrics/gates", gates), ("/retrain/status", retrain_status),
                   ("/predict/next-day", next_day), ("/metrics/upstream", upstream),
                   ("/schedule/reschedule", reschedule), ("/schedule/berth-allocation", berth_allocation)):
    app.router.routes.insert(0, APIRoute(_path, _fn, methods=["GET"]))
