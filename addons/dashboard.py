"""
v2.5: 팀 대시보드(부산항 지연 조기경보) ↔ 접안 보정 모델 연결. 신항 5개 터미널 한 파일.

기존 serving_app 파일은 고치지 않고, 같은 FastAPI 앱에 경로 두 개만 덧붙인다.
  GET /early-warning   : 대시보드 화면 (addons/early_warning.html, v2.5 파일 보존)
  GET /v25/board       : 화면이 쓰는 데이터 한 번에
나머지(/data/upload, /predict/replay, /retrain/status, /logs, /health)는 기존 API 그대로 쓴다.

업로드 파일: HJNC 엑셀 그대로, 또는 `터미널` 칸이 있는 통합 CSV (addons/terminals.py 가 만든 것).
`터미널` 칸이 없으면 전부 HJNC로 본다. 계산은 터미널별로 따로 한다 (같은 항로 최근 20척도 같은 터미널 안에서).
반입 마감이 비어 있는 배(HPNT·DGT처럼 정답을 못 만드는 터미널)는 척수에만 들어가고 예측·검증에서 빠진다.

/v25/board 가 돌려주는 것
  days     : 운영 구간(2026-05~) 하루별 실제·예측 접안 지연, 최근 21척 오차(drift_detector와 같은 계산), 그날 배.
             모두 "반입 마감(접안 10시간 전) 시점에 그때까지 접안한 배만 보고 낸 예측"이다 (과거인 척 예측)
  upcoming : as_of 기준 다음 horizon_h 시간 안에 접안 예정인 배. as_of가 과거면 실제 접안도 붙는다 (과거 시점 재현)
  recent   : as_of 전 24시간 안에 접안한 배 (지금 접안해 있는 배 포함). 각자 반입 마감 시점 예측 vs 실제
  terminals: 터미널별 전체 기간 정확도 (터미널 예정 vs 모델, 30분 안 비율)

숫자 기준
  - 접안 지연(h) = 실제 접안 − 터미널 예정(반입 마감 + 10h)
  - 최근 21척 판정 = 모델 MAE ≥ 터미널 MAE + 0.05h 또는 RMSE > 1.0h. 여기서는 표시만, 재학습은 /predict/replay 버튼
  - 배 등급: |예측 지연| 30분 미만 정상, 30분~1시간, 1~2시간, 2시간 이상 경고 (팀 초안 1/3/6시간에서 실데이터 오차 수준으로)
  - 모델은 HJNC 2025년으로 학습. PNIT·BNCT는 학습에 쓰지 않은 터미널이라 전 기간이 처음 보는 데이터
"""
import json
import os
import tempfile
import time

import pandas as pd
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute

from addons.rules import RULES, tugs_for
from data.features import GROUP_KEY, LEAD_H, VAL_END, _history_for, berthed_history, load_samples, sequence_points
from data.hjnc import read_berth_excel
from data.storage import latest_upload
from serving_app import model_loader, state
from serving_app.main import app
from serving_app.monitoring.drift_detector import ABS_RMSE_THRESHOLD, REL_TOLERANCE_HOURS
from addons import drift as dr  # v2.5: 14일 판정 창 + 평균 치우침 규칙, 재학습 28일 (import 시 기존 판정 함수에 끼워 넣음)

WINDOW_SIZE = dr.MIN_SHIPS  # 화면 호환용: 이제 '판정 최소 척수'. 창 자체는 dr.WINDOW_DAYS 일

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "early_warning.html")  # v2.6 디자인. v2.5 파일은 보존
INFO_PATH = os.path.join(HERE, "vessel_info.json")
GRADES = (0.5, 1.0, 2.0)
OPS_START = VAL_END  # 2026-05-01. HJNC 학습 2025, 검증 2026-01~04 이후가 운영(테스트) 구간
RECENT_H = 24.0

_cache: dict = {}


def _vessel_info() -> dict:
    if "info" not in _cache:
        _cache["info"] = json.load(open(INFO_PATH, encoding="utf-8"))["vessels"] if os.path.exists(INFO_PATH) else {}
    return _cache["info"]


def _ship(row: pd.Series, term: str, delay: float, actual: float | None, plan: pd.Timestamp, day0: pd.Timestamp) -> dict:
    """배 1척 → 화면 카드. eta/arr는 day0(그날 0시) 기준 시간(h)."""
    nm = str(row["선박명"]).strip()
    info = _vessel_info().get(nm.upper(), {})
    gt = info.get("gt")
    eta = (plan - day0).total_seconds() / 3600
    berthed = actual is not None
    return {"terminal": term, "name": nm, "call": row["모선항차"], "line": row["선사"], "route": row[GROUP_KEY],
            "berth": str(row["선석"]), "eta": round(eta, 3), "delay": round(delay, 3), "arr": round(eta + delay, 3),
            "actual": round(actual, 3) if berthed else None,
            "planned": plan.strftime("%Y-%m-%d %H:%M"),
            "predicted": (plan + pd.Timedelta(hours=delay)).strftime("%Y-%m-%d %H:%M"),
            "actual_berth": row["입항일시"].strftime("%Y-%m-%d %H:%M") if berthed else None,
            "working": bool(berthed and pd.isna(row["작업 완료일시"])),
            "unload": int(row["양하"]), "load": int(row["선적"]), "shift": int(row["S/H"]), "box": int(row["box"]),
            "gt": gt, "flag": info.get("flag"), "type": info.get("type"), "up": info.get("prev_port_last"),
            "up_top": info.get("prev_port_top"), "crew": info.get("crew"), "seen": info.get("last_seen"),
            "tugs": tugs_for(gt), "pilot": None if gt is None else gt >= RULES["compulsory_pilot_gt"]["value"]}


def _stats(pairs: list[tuple[float, float]]) -> dict:
    """pairs = [(실제, 예측)] → 모델·터미널 MAE, RMSE, 평균 치우침, 30분 안 비율, 드리프트 조건 (addons/drift.stats)."""
    return dr.stats(pairs)


def _split(path: str) -> dict:
    """업로드 파일 → {터미널: (표, 터미널만 담은 임시 CSV 경로)}. 기존 load_samples가 파일 경로를 받기 때문."""
    key = (path, os.path.getmtime(path))
    if _cache.get("split_key") == key:
        return _cache["split"]
    df = read_berth_excel(path)
    if "터미널" not in df.columns:
        df["터미널"] = "HJNC"
    tmp = tempfile.mkdtemp(prefix="v25_")
    out = {}
    for term, g in df.groupby("터미널", sort=False):
        p = os.path.join(tmp, f"{term}.csv")
        g.drop(columns=[c for c in ["box", "done", "plan_berth_10h", "work_h", "berth_h", "pre_h", "post_h", "prod"] if c in g])\
            .to_csv(p, index=False)
        out[term] = (read_berth_excel(p), p)
    _cache["split_key"], _cache["split"] = key, out
    return out


def _labelable(df: pd.DataFrame) -> bool:
    return bool(df["반입 마감일시"].notna().any())


def _history(parts: dict, model) -> dict:
    key = (_cache.get("split_key"), model.version, state.fallback)
    if _cache.get("hist_key") == key:
        return _cache["hist"]
    recs = []
    for term, (df, p) in parts.items():
        if not _labelable(df):
            continue
        idx = df.set_index("모선항차", drop=False)
        samples = load_samples(p)
        preds = model.predict_many([s["sequence"] for s in samples]) if samples else []
        for s, pr in zip(samples, preds):
            row = idx.loc[s["call_id"]]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            recs.append({"term": term, "s": s, "row": row, "p": float(pr), "a": s["actual_correction"]})
    recs.sort(key=lambda r: r["s"]["planned_berth"])

    terms = {}
    for term in parts:
        rs = [r for r in recs if r["term"] == term]
        st = _stats([(r["a"], r["p"]) for r in rs])
        ops = _stats([(r["a"], r["p"]) for r in rs if r["s"]["planned_berth"] >= OPS_START])
        df = parts[term][0]
        terms[term] = {"labelable": _labelable(df), "calls": int(len(df)),
                       "first": str(df["입항일시"].min())[:10], "last": str(df["입항일시"].max())[:10],
                       "all": st, "ops": ops, "trained_on": term == "HJNC"}

    by_day: dict = {}
    for r in recs:
        by_day.setdefault(r["s"]["planned_berth"].normalize(), []).append(r)
    counts = {}
    for term, (df, _) in parts.items():
        for d0, n in df["입항일시"].dt.normalize().value_counts().items():
            counts.setdefault(d0, {})[term] = int(n)

    days = []
    for d0 in sorted(by_day):
        rows = by_day[d0]
        if d0 < OPS_START:
            continue
        # 판정 창: 그날 끝 기준 최근 14일 (터미널별·전체)
        lo, hi = d0 + pd.Timedelta(days=1) - pd.Timedelta(days=dr.WINDOW_DAYS), d0 + pd.Timedelta(days=1)
        win_all = [r for r in recs if lo <= r["s"]["planned_berth"] < hi]
        win_t = {}
        for r in win_all:
            win_t.setdefault(r["term"], []).append(r)
        by_term = {}
        for term in {r["term"] for r in rows} | set(win_t):
            tr = [r for r in rows if r["term"] == term]
            by_term[term] = {"n_ships": len(tr), "win": _stats([(r["a"], r["p"]) for r in win_t.get(term, [])]),
                             **({"actual": _mean([r["a"] for r in tr]), "pred": _mean([r["p"] for r in tr]),
                                 "model_abs": _mean([abs(r["a"] - r["p"]) for r in tr]),
                                 "term_abs": _mean([abs(r["a"]) for r in tr])} if tr else {})}
        days.append({"iso": d0.strftime("%Y-%m-%d"), "dow": int((d0.dayofweek + 1) % 7), "n_ships": len(rows),
                     "actual": _mean([r["a"] for r in rows]), "pred": _mean([r["p"] for r in rows]),
                     "model_abs": _mean([abs(r["a"] - r["p"]) for r in rows]), "term_abs": _mean([abs(r["a"]) for r in rows]),
                     "win": _stats([(r["a"], r["p"]) for r in win_all]), "by_term": by_term,
                     "calls_by_term": counts.get(d0, {}),
                     "ships": [_ship(r["row"], r["term"], r["p"], r["a"], r["s"]["planned_berth"], d0) for r in rows]})
    out = {"days": days, "terminals": terms}
    _cache["hist_key"], _cache["hist"] = key, out
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def _plan(df: pd.DataFrame) -> pd.Series:
    """반입 마감 시점의 터미널 예정 접안. 반입 마감이 없으면(예정 배) 지금 적힌 접안 예정."""
    return pd.to_datetime(df["plan_berth_10h"]).fillna(pd.to_datetime(df["입항일시"]))


def _predict_rows(df: pd.DataFrame, rows: pd.DataFrame, as_ofs: list, model) -> list:
    """각 배를 as_of 시점까지 접안한 배만 보고 예측 (미래 정보 없음). 이력이 20척 안 되면 None."""
    hist = berthed_history(df)
    seqs, keep = [], []
    for (_, r), t in zip(rows.iterrows(), as_ofs):
        h = _history_for(hist, r[GROUP_KEY], t)
        if h is not None:
            seqs.append(sequence_points(h))
            keep.append(r)
    if not seqs:
        return []
    preds = [0.0] * len(seqs) if state.fallback and state.fallback_target == "terminal" else model.predict_many(seqs)
    return list(zip(keep, preds))


def _window(parts: dict, t0: pd.Timestamp, horizon_h: float, model):
    """upcoming: 접안 예정이 (t0, t0+h] 이고 t0에 아직 접안 전인 배 → t0 시점 예측 (+ 실제가 있으면 붙임)
       recent  : (t0−24h, t0] 에 접안한 배 → 각자 반입 마감 시점 예측 vs 실제"""
    d0, up, rec, congestion = t0.normalize(), [], [], {}
    for term, (df, _) in parts.items():
        plan = _plan(df)
        berthed = df["작업 시작일시"].notna()
        congestion[term] = int(((plan > t0) & (plan <= t0 + pd.Timedelta(hours=horizon_h))).sum())
        if not _labelable(df):
            continue
        m = (plan > t0) & (plan <= t0 + pd.Timedelta(hours=horizon_h)) & (~berthed | (df["입항일시"] > t0))
        rows = df[m]
        for r, p in _predict_rows(df, rows, [t0] * len(rows), model):
            act = (r["입항일시"] - _plan(r.to_frame().T).iloc[0]).total_seconds() / 3600 if pd.notna(r["작업 시작일시"]) else None
            up.append(_ship(r, term, float(p), act, _plan(r.to_frame().T).iloc[0], d0))
        m2 = berthed & (df["입항일시"] > t0 - pd.Timedelta(hours=RECENT_H)) & (df["입항일시"] <= t0) & df["plan_berth_10h"].notna()
        rows = df[m2]
        for r, p in _predict_rows(df, rows, list(rows["plan_berth_10h"] - pd.Timedelta(hours=LEAD_H)), model):
            act = (r["입항일시"] - r["plan_berth_10h"]).total_seconds() / 3600
            if abs(act) <= 48:
                rec.append(_ship(r, term, float(p), act, r["plan_berth_10h"], d0))
    up.sort(key=lambda s: s["eta"])
    rec.sort(key=lambda s: s["eta"])
    return up, rec, congestion


def board(as_of: str | None = None, horizon_h: float = 48.0):
    try:
        path = latest_upload()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    model = model_loader.get_model()
    t_start = time.time()
    parts = _split(path)
    hist = _history(parts, model)
    t0 = pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).floor("min")
    up, rec, congestion = _window(parts, t0, horizon_h, model)
    last_known = max(df["작업 시작일시"].max() for df, _ in (v for v in parts.values()))
    version = ("fallback:" + state.fallback_target) if state.fallback else model.version
    return {"file": os.path.basename(path), "model_version": version, "as_of": t0.strftime("%Y-%m-%d %H:%M"),
            "horizon_h": horizon_h, "recent_h": RECENT_H, "replay": bool(t0 < pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None) - pd.Timedelta(minutes=30)),
            "data_last_berth": None if pd.isna(last_known) else last_known.strftime("%Y-%m-%d %H:%M"),
            "upcoming": up, "upcoming_stats": _stats([(s["actual"], s["delay"]) for s in up if s["actual"] is not None]),
            "recent": rec, "recent_stats": _stats([(s["actual"], s["delay"]) for s in rec]),
            "congestion_next": congestion, "terminals": hist["terminals"], "days": hist["days"],
            "ops_start": OPS_START.strftime("%Y-%m-%d"), "grades_h": GRADES, "window": WINDOW_SIZE, "window_days": dr.WINDOW_DAYS, "retrain_days": dr.RETRAIN_DAYS,
            "gate": {"rel_tolerance_h": REL_TOLERANCE_HOURS, "abs_rmse_h": ABS_RMSE_THRESHOLD, "bias_h": round(dr.BIAS_H, 3), "window_days": dr.WINDOW_DAYS, "min_ships": dr.MIN_SHIPS},
            "elapsed_s": round(time.time() - t_start, 2)}


def page():
    return FileResponse(HTML, media_type="text/html")


# StaticFiles가 "/"에 마운트돼 있어 뒤에 붙이면 가려진다 → 라우트 목록 맨 앞에 넣는다
for _path, _fn in (("/v25/board", board), ("/early-warning", page)):
    app.router.routes.insert(0, APIRoute(_path, _fn, methods=["GET"]))


# v2.5 팀 화면 연결 API (/metrics/*, /schedule/*, /predict/next-day·/retrain/status 확장)
import addons.api  # noqa: E402,F401

# v2.5 관리자 콘솔 (/admin, /admin/api/*)
import addons.admin  # noqa: E402,F401

# v2.5 손해액·추천 행동 (/v25/impact), AIS 앙상블 (/v25/ais-eta, /v25/upstream-queue) — 없으면 건너뜀
for _m in ("addons.loss", "addons.impact", "addons.ais_eta", "addons.perf", "addons.routes"):
    try:
        __import__(_m)
    except ImportError as _e:
        print(f"[v2.5] {_m} 건너뜀: {_e}")

# v2.6 예측 시점별 정확도 (/v25/lead-accuracy, /v25/lead-history)
try:
    import addons.leadtime  # noqa: E402,F401
except ImportError as _e:
    print(f"[v2.6] leadtime 건너뜀: {_e}")
