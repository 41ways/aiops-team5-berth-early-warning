"""
v2.5: 실시간 AIS로 접안 직전 예측 보정 (앙상블 v0). 기존 파일은 고치지 않고 경로 하나를 앞에 끼운다.

  GET /v25/upstream-queue?as_of=   앞 항만(상하이·닝보-저우산) + 부산신항 묘박지 대기 척수, 최근 7일 6시간 간격
  GET /v25/ais-eta?as_of=&horizon_h=48
    /v25/board 의 upcoming 배마다
      - AIS 매칭(배 이름 → MMSI), as_of 직전 마지막 위치, 신항까지 거리, 최근 6시간 평균 속력, 마지막 수신 경과
      - 물리 ETA(거리 ÷ 속력), 선원 입력 ETA, 묘박지 대기 척수
      - 앙상블 접안 예측 = 모델 A + w × (물리 ETA − 모델 A)
      - 예상 출항(선석 점유 끝)과 터미널 예정 출항 대비 작업 지연

실행 (dashboard.py 를 고치지 않고 켜기)
  uvicorn addons.ais_eta:app --port 8079
AIS 데이터: collectors/ais_collector.py 가 쓰는 data/ais/positions_YYYYMMDD.csv, static_YYYYMMDD.csv (UTC).
  다른 폴더를 보려면 AIS_DIR 환경변수. 파일이 없으면 모든 배 w=0, 모델 A 그대로 (에러 아님).

시간대: AIS CSV는 UTC, 스케줄·as_of는 한국 시각(KST, tz 없는 값). load_positions 가 t_kst(KST, tz 없음)를 만든다.
가중치 w(물리 쪽)는 검증 전 규칙값(v0)이다. AIS가 쌓이면 리드타임별로 검증 구간에서 다시 정할 것.
물리 ETA는 신항 입구(PORT_REF) 도착 시각이라 도선·접안 소요가 빠져 있다 → 가중치가 작고 거리가 가까울 때만 크다.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from fastapi.routing import APIRoute

from addons import dashboard as dv
from data.ais_features import (PORT_REF, anchorage_queue, declared_eta, haversine_nm, load_positions, match_mmsi,
                               physics_eta, track_sequence)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_AIS_DIR = os.path.join(ROOT, "data", "ais")
STALE_MIN = 120          # 마지막 수신이 2시간 넘으면 w=0
MIN_SOG = 3.0            # 6시간 평균 속력 3노트 미만(정박·표류)이면 w=0
MAX_GAP_H = 12.0         # 물리 ETA가 모델 A와 12시간 넘게 다르면 매칭 오류로 보고 w=0
DIST_W = ((30, 0.75), (100, 0.5), (200, 0.25))  # 거리(해리) 이하 → w
POS_DAYS_BACK = 1        # as_of(UTC) 날짜 기준 하루 전 파일까지 (track_sequence 20시간 창)
STATIC_DAYS_BACK = 14    # 이름 매칭용 정적 정보는 2주 치
WEIGHT_NOTE = "가중치는 검증 전 규칙값(v0)"
HOW_TO = ("aisstream.io에서 GitHub 로그인으로 키 발급 → 터미널에서 AISSTREAM_API_KEY=키 python collectors/ais_collector.py 실행 "
          "→ data/ais/에 쌓이면 자동 반영")

_cache: dict = {}


def _ais_dir(ais_dir: str | None = None) -> str:
    return ais_dir or os.getenv("AIS_DIR") or DEFAULT_AIS_DIR


def _file_day(p: str):
    m = re.search(r"_(\d{8})\.csv$", p)
    return pd.Timestamp(m.group(1)) if m else None


def _pick(files: list, as_of_kst: pd.Timestamp, days_back: int) -> list:
    """파일 이름 날짜(UTC)가 [as_of(UTC) 날짜 − days_back, as_of(UTC) 날짜] 인 것만."""
    d_utc = (as_of_kst - pd.Timedelta(hours=9)).normalize()
    out = []
    for p in files:
        d = _file_day(p)
        if d is not None and d_utc - pd.Timedelta(days=days_back) <= d <= d_utc:
            out.append(p)
    return sorted(out)


def _read(kind: str, paths: list, slot: str | None = None):
    """같은 파일 묶음이면 캐시 (수집기가 계속 덧붙이므로 수정 시각까지 키에 넣는다). slot: 캐시 칸 이름."""
    if not paths:
        return None
    slot = slot or kind
    key = (kind, tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in paths))
    if _cache.get(slot + "_key") == key:
        return _cache[slot]
    if kind == "pos":
        df = load_positions(paths)
    else:
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        df["time_utc"] = pd.to_datetime(df["time_utc"].astype(str).str.slice(0, 19), utc=True, errors="coerce")
        df = df.dropna(subset=["time_utc", "mmsi"])
        df["t_kst"] = df["time_utc"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    df = df[pd.to_numeric(df["mmsi"], errors="coerce").notna()].copy()
    df["mmsi"] = df["mmsi"].astype("int64")
    _cache[slot + "_key"], _cache[slot] = key, df
    return df


def load_ais(as_of_kst: pd.Timestamp, ais_dir: str | None = None) -> dict:
    """as_of 시점에 쓸 수 있는 AIS만 (as_of 이후 신호는 버린다)."""
    d = _ais_dir(ais_dir)
    pos_all = sorted(glob.glob(os.path.join(d, "positions_*.csv")))
    sta_all = sorted(glob.glob(os.path.join(d, "static_*.csv")))
    out = {"dir": d, "files": [os.path.basename(p) for p in pos_all + sta_all], "pos": None, "static": None, "error": None}
    try:
        pos = _read("pos", _pick(pos_all, as_of_kst, POS_DAYS_BACK))
        sta = _read("static", _pick(sta_all, as_of_kst, STATIC_DAYS_BACK))
        if pos is not None:
            pos = pos[pos["t_kst"] <= as_of_kst]
            out["pos"] = pos if len(pos) else None
        if sta is not None:
            sta = sta[sta["t_kst"] <= as_of_kst]
            out["static"] = sta if len(sta) else None
    except Exception as e:  # 파일 형식이 깨져도 화면은 모델 A로 계속 나가게
        out["error"] = f"{type(e).__name__}: {e}"
        out["pos"] = out["static"] = None
    return out


def ais_weight(has_ais: bool, last_seen_min, sog_6h, dist_nm, phys: pd.Timestamp | None, model_a: pd.Timestamp):
    """물리 ETA 쪽 가중치 w와 이유. 규칙값 v0 (검증 전)."""
    if not has_ais:
        return 0.0, "AIS 없음 (매칭 실패 또는 as_of 직전 20시간 위치 없음)"
    if last_seen_min is None or last_seen_min > STALE_MIN:
        return 0.0, f"마지막 수신 {last_seen_min:.0f}분 전 (> {STALE_MIN}분)" if last_seen_min is not None else "마지막 수신 시각 없음"
    if sog_6h is None or not np.isfinite(sog_6h) or sog_6h < MIN_SOG:
        return 0.0, f"최근 6시간 평균 속력 {sog_6h:.1f}노트 (< {MIN_SOG:g}노트, 정박·대기)" if sog_6h is not None else "속력 없음"
    if phys is None:
        return 0.0, "물리 ETA 계산 불가"
    gap = abs((phys - model_a).total_seconds()) / 3600
    if gap > MAX_GAP_H:
        return 0.0, f"물리 ETA가 모델 A와 {gap:.1f}시간 차이 (> {MAX_GAP_H:g}시간, 매칭 오류 의심)"
    for lim, w in DIST_W:
        if dist_nm <= lim:
            return w, f"신항까지 {dist_nm:.0f}해리 (≤ {lim}해리) → w={w}"
    return 0.0, f"신항까지 {dist_nm:.0f}해리 (> {DIST_W[-1][0]}해리, 너무 멀다)"


def _fmt(t):
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def _ais_for(ship: dict, mmsi, ais: dict, t0: pd.Timestamp, queue) -> dict:
    """배 1척 AIS 특성. 모두 KST 기준(t0도 KST)."""
    a = {"mmsi": mmsi, "last_fix_kst": None, "last_fix_utc": None, "lat": None, "lon": None, "dist_nm": None,
         "sog_6h_kn": None, "last_seen_min": None, "coverage_20h": None, "physics_eta": None, "declared_eta": None,
         "destination": None, "anchorage_queue": queue}
    pos = ais["pos"]
    if mmsi is None or pos is None:
        return a
    g = pos[pos["mmsi"] == mmsi]
    if g.empty:
        return a
    last = g.iloc[-1]  # load_positions 가 mmsi, t_kst 순으로 정렬
    a.update({"last_fix_kst": _fmt(last["t_kst"]), "last_fix_utc": last["time_utc"].strftime("%Y-%m-%dT%H:%MZ"),
              "lat": round(float(last["lat"]), 4), "lon": round(float(last["lon"]), 4),
              "dist_nm": round(float(haversine_nm(last["lat"], last["lon"], *PORT_REF)), 1),
              "last_seen_min": round((t0 - last["t_kst"]).total_seconds() / 60, 1)})
    seq, q = track_sequence(pos, mmsi, t0)
    a["coverage_20h"] = q["coverage"]
    if seq is not None:
        s = float(np.nanmean(seq[-6:, 1]))
        a["sog_6h_kn"] = round(s, 1) if np.isfinite(s) else None
        a["_phys"] = physics_eta(seq, last["t_kst"])  # 마지막 수신 시각 + 그때 거리 ÷ 속력 (as_of 기준이면 수신 공백만큼 늦게 나온다)
        a["physics_eta"] = _fmt(a["_phys"])
    sta = ais["static"]
    if sta is not None:
        sr = sta[sta["mmsi"] == mmsi].sort_values("t_kst")
        if len(sr):
            r = sr.iloc[-1]
            a["declared_eta"] = _fmt(declared_eta(r, t0))
            a["destination"] = None if pd.isna(r.get("destination")) else str(r.get("destination"))
    return a


def enrich(up: list, t0: pd.Timestamp, ais_dir: str | None = None, estimators: dict | None = None,
           parts: dict | None = None) -> dict:
    """upcoming 목록 → AIS 특성·앙상블·작업 지연. 서버 없이도 부를 수 있게 분리 (단위 확인용)."""
    ais = load_ais(t0, ais_dir)
    available = ais["pos"] is not None
    mm = {}
    if ais["static"] is not None and up:
        mm = match_mmsi(ais["static"], pd.DataFrame({"선박명": [s["name"] for s in up]}))
    queue = anchorage_queue(ais["pos"], t0) if available else None
    out = []
    for s in up:
        plan = pd.Timestamp(s["planned"])
        model_a = plan + pd.Timedelta(hours=float(s["delay"]))  # "predicted" 문자열은 분 단위로 잘려 있어 delay로 다시 만든다
        mmsi = mm.get(s["name"])
        a = _ais_for(s, mmsi, ais, t0, queue)
        phys = a.pop("_phys", None)
        has = available and a["last_fix_kst"] is not None
        w, why = ais_weight(bool(has), a["last_seen_min"], a["sog_6h_kn"], a["dist_nm"], phys, model_a)
        if not available:
            why = "AIS 파일 없음" if not ais["files"] else ("AIS 파일 읽기 실패" if ais["error"] else "as_of 직전 AIS 위치 없음")
        elif mmsi is None:
            why = "AIS 이름 매칭 실패"
        ens = model_a + (phys - model_a) * w if (w > 0 and phys is not None) else model_a
        rec = {"terminal": s["terminal"], "name": s["name"], "call": s["call"], "route": s["route"], "berth": s["berth"],
               "planned": s["planned"], "predicted_model_a": s["predicted"], "delay_model_a_h": s["delay"],
               "ais": a, "w": w, "reason": why, "weight_note": WEIGHT_NOTE,
               "ensemble": _fmt(ens), "ensemble_delay_h": round((ens - plan).total_seconds() / 3600, 3),
               "actual_berth": s.get("actual_berth"), "actual_delay_h": s.get("actual")}
        rec.update(_work(s, ens, t0, estimators, parts))
        out.append(rec)
    last_fix = ais["pos"]["time_utc"].max().strftime("%Y-%m-%dT%H:%M:%SZ") if available else None
    return {"ais_available": available, "ais_dir": os.path.relpath(ais["dir"], ROOT) if ais["dir"].startswith(ROOT) else ais["dir"],
            "ais_files": ais["files"], "ais_error": ais["error"], "last_fix_utc": last_fix,
            "matched_n": sum(1 for r in out if r["ais"]["mmsi"] is not None), "total_n": len(out),
            "used_n": sum(1 for r in out if r["w"] > 0), "anchorage_queue": queue,
            "how_to_enable": HOW_TO, "weight_note": WEIGHT_NOTE,
            "weight_rule": {"stale_min": STALE_MIN, "min_sog_kn": MIN_SOG, "max_gap_vs_model_a_h": MAX_GAP_H,
                            "dist_nm_to_w": [{"max_nm": d, "w": w} for d, w in DIST_W], "else_w": 0.0,
                            "formula": "ensemble = 모델 A + w × (물리 ETA − 모델 A)"},
            "vessels": out}


def _work(s: dict, berth: pd.Timestamp, t0: pd.Timestamp, estimators, parts) -> dict:
    """예상 출항 = 앙상블 접안 + WorkTimeEstimator(사전·작업·사후 시간). 대시보드에는 출항 예측이 없어 여기서 계산."""
    na = {"predicted_departure": None, "terminal_planned_departure": None, "work_delay_h": None, "predicted_work_h": None}
    try:
        e = (estimators or {}).get(s["terminal"])
        if e is None:
            return na
        dep = e.departure(berth, s["box"], s["route"], t0)
        tpd = None
        if parts and s["terminal"] in parts:
            df = parts[s["terminal"]][0]
            row = df[df["모선항차"] == s["call"]]
            if len(row) and pd.notna(row.iloc[0]["출항일시"]):
                tpd = pd.Timestamp(row.iloc[0]["출항일시"])
        return {"predicted_departure": _fmt(dep), "terminal_planned_departure": _fmt(tpd),
                "work_delay_h": round((dep - tpd).total_seconds() / 3600, 2) if tpd is not None else None,
                "predicted_work_h": round(e.work_hours(s["box"], s["route"], t0), 1)}
    except Exception:
        return na


def ais_eta(as_of: str | None = None, horizon_h: float = 48.0):
    b = dv.board(as_of, horizon_h)
    t0 = pd.Timestamp(b["as_of"])
    est, parts = None, None
    try:
        from addons.api import _estimators
        from data.storage import latest_upload
        est, parts = _estimators(), dv._split(latest_upload())
    except Exception:
        pass
    res = enrich(b["upcoming"], t0, estimators=est, parts=parts)
    return {"as_of": b["as_of"], "horizon_h": horizon_h, "model_version": b["model_version"], "file": b["file"],
            "timezone": "출력 시각은 모두 KST(YYYY-MM-DD HH:MM), *_utc 만 UTC",
            "physics_note": "물리 ETA = 마지막 수신 시각 + 그때 신항 입구까지 거리 ÷ 최근 6시간 평균 속력 (도선·접안 소요 미포함)",
            "work_note": "predicted_departure = 앙상블 접안 + 사전·작업·사후 시간(serving_app/schedule.py WorkTimeEstimator), "
                         "work_delay_h = predicted_departure − 터미널 예정 출항",
            **res}


# ---------------------------------------------------------------------------------------------
# 앞 항만 대기 척수 (실시간 AIS). anchorage_queue 와 같은 방식: 반경 안 + 속력 < 1노트 + 최근 2시간 창의 MMSI 수.
# 좌표·반경은 모두 대략값 (해도 묘박지 경계가 아니라 그 부근을 덮는 원). 상하이는 양산·우쑹 두 원의 합집합.
# ---------------------------------------------------------------------------------------------
from collectors.ais_collector import DEFAULT_BOX  # noqa: E402  (websockets는 run() 안에서만 import → 서버에서 안전)
from data.ais_features import ANCHORAGE  # noqa: E402

QUEUE_WINDOW_H = 2.0
QUEUE_SOG = 1.0
SERIES_DAYS = 7
SERIES_STEP_H = 6
UPSTREAM_PORTS = [
    {"port": "SHANGHAI", "label": "상하이 (양산·우쑹 묘박지 부근)",
     "zones": [{"name": "양산항 외곽 묘박지 부근", "center": (30.55, 122.20), "radius_nm": 12},
               {"name": "우쑹·장강 하구 묘박지 부근", "center": (31.30, 121.85), "radius_nm": 15}]},
    {"port": "NINGBO_ZHOUSHAN", "label": "닝보-저우산 묘박지 부근",
     "zones": [{"name": "베이룬·저우산 묘박지 부근", "center": (29.90, 122.15), "radius_nm": 15}]},
    {"port": "BUSAN_NEW", "label": "부산신항 묘박지 (data/ais_features.ANCHORAGE)",
     "zones": [{"name": "신항 외곽 묘박지 부근", "center": ANCHORAGE[0], "radius_nm": ANCHORAGE[1]}]},
    {"port": "QINGDAO", "label": "칭다오", "zones": [{"name": "칭다오항 부근", "center": (36.05, 120.35), "radius_nm": 15}]},
]
QUEUE_HOW_TO = (HOW_TO + ". 칭다오를 보려면 수집 범위를 넓혀야 하는데 수집기에는 범위 환경변수(AISSTREAM_BOX 등)가 없다 "
                "(환경변수는 AIS_OUT_DIR, AIS_MMSI 뿐) → collectors/ais_collector.py 의 DEFAULT_BOX 서쪽 경도를 120.0 쯤으로 바꿔야 함")


def _in_box(lat, lon) -> bool:
    return any(b[0][0] <= lat <= b[1][0] and b[0][1] <= lon <= b[1][1] for b in DEFAULT_BOX)


def _zone_mask(pos: pd.DataFrame, zones: list) -> pd.Series:
    m = pd.Series(False, index=pos.index)
    for z in zones:
        m |= haversine_nm(pos["lat"], pos["lon"], *z["center"]) <= z["radius_nm"]
    return m


def _count(stopped: pd.DataFrame, t: pd.Timestamp) -> int:
    g = stopped[(stopped["t_kst"] <= t) & (stopped["t_kst"] > t - pd.Timedelta(hours=QUEUE_WINDOW_H))]
    return int(g["mmsi"].nunique())


def upstream_queue_at(as_of: str | None = None, ais_dir: str | None = None):
    t0 = pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).floor("min")
    d = _ais_dir(ais_dir)
    pos_all = sorted(glob.glob(os.path.join(d, "positions_*.csv")))
    base = {"as_of": t0.strftime("%Y-%m-%d %H:%M"), "metric": f"반경 안에서 속력 < {QUEUE_SOG:g}노트로 거의 멈춘 배 수 "
            f"(최근 {QUEUE_WINDOW_H:g}시간 창, MMSI 중복 제거, 선종 구분 없음)",
            "coords_note": "대략 좌표 (실제 묘박지 경계 아님, 부근을 덮는 원)", "collector_box": DEFAULT_BOX,
            "series_step_h": SERIES_STEP_H, "series_null": "그 창(2시간)에 수신 기록이 하나도 없으면 null (수집기 꺼짐)", "series_days": SERIES_DAYS, "how_to_enable": QUEUE_HOW_TO,
            "ais_dir": os.path.relpath(d, ROOT) if d.startswith(ROOT) else d, "ais_files": [os.path.basename(p) for p in pos_all],
            "timezone": "t_kst 는 KST, last_fix_utc 만 UTC"}
    pos, err = None, None
    try:
        pos = _read("pos", _pick(pos_all, t0, SERIES_DAYS + 1), slot="pos_queue")
        if pos is not None:
            pos = pos[pos["t_kst"] <= t0]
            pos = pos if len(pos) else None
    except Exception as e:
        err, pos = f"{type(e).__name__}: {e}", None
    ports = []
    if pos is not None:
        first = pos["t_kst"].min()
        steps = pd.date_range(t0 - pd.Timedelta(days=SERIES_DAYS), t0, freq=f"{SERIES_STEP_H}h")
        steps = [t for t in steps if t >= first]  # 파일에 있는 만큼만
        if not steps or steps[-1] != t0:
            steps = list(steps) + [t0]
        slow = pos[pos["sog"] < QUEUE_SOG]
        ts = np.sort(pos["t_kst"].values)
        # 창 안에 수신 기록이 하나도 없으면(수집기 꺼짐) 0이 아니라 null
        live = {t: np.searchsorted(ts, np.datetime64(t), "right") > np.searchsorted(ts, np.datetime64(t - pd.Timedelta(hours=QUEUE_WINDOW_H)), "right")
                for t in steps}
    for p in UPSTREAM_PORTS:
        inbox = all(_in_box(*z["center"]) for z in p["zones"])
        rec = {"port": p["port"], "label": p["label"], "approx_coords": True, "in_collector_box": inbox,
               "zones": [{"name": z["name"], "center": list(z["center"]), "radius_nm": z["radius_nm"]} for z in p["zones"]],
               "queue_now": None, "series": []}
        if not inbox:
            rec["note"] = "수집 범위 밖, AISSTREAM_BOX 넓혀야 (수집기에 그 환경변수는 아직 없음 → DEFAULT_BOX 수정 필요)"
        elif pos is not None:
            stopped = slow[_zone_mask(slow, p["zones"])]
            rec["queue_now"] = _count(stopped, t0)
            rec["series"] = [{"t_kst": t.strftime("%Y-%m-%d %H:%M"), "n": _count(stopped, t) if live[t] else None} for t in steps]
        ports.append(rec)
    if pos is not None:  # 부산은 data/ais_features.anchorage_queue 와 같은 값인지 그대로 재사용해 맞춘다
        ports[2]["queue_now_check"] = anchorage_queue(pos, t0, QUEUE_WINDOW_H)
    return {**base, "available": pos is not None, "error": err,
            "last_fix_utc": pos["time_utc"].max().strftime("%Y-%m-%dT%H:%M:%SZ") if pos is not None else None,
            "first_fix_kst": pos["t_kst"].min().strftime("%Y-%m-%d %H:%M") if pos is not None else None, "ports": ports}


def upstream_queue(as_of: str | None = None):
    """라우트용 (ais_dir를 쿼리로 열지 않는다. 폴더는 AIS_DIR 환경변수)."""
    return upstream_queue_at(as_of)


app = dv.app
for _path, _fn in (("/v25/ais-eta", ais_eta), ("/v25/upstream-queue", upstream_queue)):
    app.router.routes.insert(0, APIRoute(_path, _fn, methods=["GET"]))
