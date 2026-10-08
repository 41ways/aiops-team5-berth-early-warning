"""
v2.6: 예측 시점별 정확도 (접안 48시간 전 / 24시간 전 / 반입 마감(10시간 전) / 6시간 전 / 최종(1시간 전)).

  GET /v25/lead-accuracy?terminal=ALL|HJNC|PNIT|BNCT
  GET /v25/lead-history?call_id=   한 배의 시점별 예측 (접안 예정 배도 지난 시점은 다시 계산)

방법 (미래 정보 없음)
  - 대상: 운영(테스트) 구간 2026-05~ 에 작업이 끝난 배. 정답 = 실제 접안 − 터미널 예정(반입 마감 + 10h)
  - 시점 L 시간 전 예측 = 그 시각(예정 접안 − L)까지 접안한 배의 이력(같은 항로 최근 20척)만 넣은 모델 예측
  - 터미널 예정 기준은 반입 마감 시점 값 하나뿐이다. 48·24시간 전에 터미널이 갖고 있던 예정은 기록이 남지 않아(지난 배는 최종값으로 덮어씀)
    그 두 시점의 "터미널 예정 오차"는 계산하지 않고, 모델 오차도 "기준 예정은 반입 마감 시점 값"이라는 표시를 붙인다
    → data/snapshots 에 시점별 엑셀이 쌓이면 scripts/plan_history.py 로 진짜 리드타임별 예정 오차를 낼 수 있다
  - 반입 마감(10시간 전) 이후 시점(6시간 전, 최종 1시간 전)은 예정은 그대로, 이력만 더 본다
"""
import threading
import time

import pandas as pd
from fastapi.routing import APIRoute

from addons import dashboard as dv
from data.features import GROUP_KEY, LEAD_H, VAL_END, _history_for, berthed_history, load_samples, sequence_points
from data.storage import latest_upload
from serving_app import model_loader
from serving_app.main import app

LEADS = [("48h", 48.0, False), ("24h", 24.0, False), ("10h", LEAD_H, True), ("6h", 6.0, True), ("final", 1.0, True)]  # (키, 시간, 터미널 예정이 그 시점에 알려져 있었나)
LABEL = {"48h": "48시간 전", "24h": "24시간 전", "10h": "반입 마감 (10시간 전)", "6h": "6시간 전", "final": "최종 (1시간 전)"}
_cache: dict = {}
_lock = threading.Lock()  # 화면이 터미널 4개를 동시에 물어도 계산은 한 번만


def _stats(errs, base):
    n = len(errs)
    if not n:
        return {"n": 0}
    return {"n": n, "mae_h": round(sum(abs(e) for e in errs) / n, 3), "in30": round(sum(abs(e) <= 0.5 for e in errs) / n, 3),
            "in60": round(sum(abs(e) <= 1.0 for e in errs) / n, 3), "bias_h": round(sum(errs) / n, 3),
            "terminal_mae_h": round(sum(abs(b) for b in base) / n, 3) if base else None,
            "terminal_in30": round(sum(abs(b) <= 0.5 for b in base) / n, 3) if base else None}


def _compute():
    with _lock:
        return _compute_locked()


def _compute_locked():
    path = latest_upload()
    model = model_loader.get_model()
    parts = dv._split(path)
    key = (dv._cache.get("split_key"), model.version)
    if _cache.get("key") == key:
        return _cache["val"]
    t = time.time()
    rows = []
    for term, (df, p) in parts.items():
        if not dv._labelable(df):
            continue
        hist = berthed_history(df)
        samples = [s for s in load_samples(p) if s["planned_berth"] >= VAL_END]
        for lk, lh, _ in LEADS:
            seqs, keep = [], []
            for s in samples:
                h = _history_for(hist, s["route"], s["planned_berth"] - pd.Timedelta(hours=lh))
                if h is not None:
                    seqs.append(sequence_points(h))
                    keep.append(s)
            preds = model.predict_many(seqs) if seqs else []
            for s, pr in zip(keep, preds):
                rows.append({"term": term, "lead": lk, "call": s["call_id"], "a": s["actual_correction"], "p": float(pr)})
    out = {}
    for term in ["ALL"] + [t for t in parts if dv._labelable(parts[t][0])]:
        rs = [r for r in rows if term == "ALL" or r["term"] == term]
        out[term] = []
        for lk, lh, known in LEADS:
            sel = [r for r in rs if r["lead"] == lk]
            st = _stats([r["a"] - r["p"] for r in sel], [r["a"] for r in sel] if known else None)
            out[term].append({"lead": lk, "label": LABEL[lk], "hours_before": lh, "plan_known": known, **st})
    val = {"by_terminal": out, "period_from": VAL_END.strftime("%Y-%m-%d"), "model_version": model.version,
           "elapsed_s": round(time.time() - t, 1),
           "note": "48·24시간 전의 터미널 예정은 기록이 없어 오차를 계산하지 않음. 그 두 시점의 모델 오차도 기준 예정을 반입 마감 시점 값으로 썼으므로 "
                   "실제보다 좋게 나올 수 있다. 오늘부터 받는 시점별 엑셀(data/snapshots)이 쌓이면 바로잡는다"}
    _cache["key"], _cache["val"] = key, val
    return val


def lead_accuracy(terminal: str = "ALL"):
    v = _compute()
    return {**{k: v[k] for k in ("period_from", "model_version", "note", "elapsed_s")}, "terminal": terminal,
            "leads": v["by_terminal"].get(terminal, [])}


def lead_history(call_id: str):
    """한 배의 시점별 예측. 이미 지난 시점만 계산 (as_of 이후 시점은 '아직')."""
    path = latest_upload()
    model = model_loader.get_model()
    now = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    for term, (df, _) in dv._split(path).items():
        r = df[df["모선항차"] == call_id]
        if r.empty:
            continue
        r = r.iloc[0]
        plan = r["plan_berth_10h"] if pd.notna(r["plan_berth_10h"]) else r["입항일시"]
        hist = berthed_history(df)
        out = []
        for lk, lh, known in LEADS:
            t0 = plan - pd.Timedelta(hours=lh)
            if t0 > now:
                out.append({"lead": lk, "label": LABEL[lk], "as_of": t0.strftime("%Y-%m-%d %H:%M"), "status": "아직"})
                continue
            h = _history_for(hist, r[GROUP_KEY], t0)
            if h is None:
                out.append({"lead": lk, "label": LABEL[lk], "status": "이력 부족"})
                continue
            d = float(model.predict_many([sequence_points(h)])[0])
            out.append({"lead": lk, "label": LABEL[lk], "as_of": t0.strftime("%Y-%m-%d %H:%M"), "status": "계산",
                        "predicted_delay_h": round(d, 3), "predicted_berth": (plan + pd.Timedelta(hours=d)).strftime("%Y-%m-%d %H:%M")})
        done = pd.notna(r["작업 시작일시"])
        return {"call_id": call_id, "terminal": term, "planned": plan.strftime("%Y-%m-%d %H:%M"),
                "actual_berth": r["입항일시"].strftime("%Y-%m-%d %H:%M") if done else None,
                "actual_delay_h": round((r["입항일시"] - plan).total_seconds() / 3600, 3) if done else None, "leads": out}
    return {"call_id": call_id, "leads": []}


for _path, _fn in (("/v25/lead-accuracy", lead_accuracy), ("/v25/lead-history", lead_history)):
    app.router.routes.insert(0, APIRoute(_path, _fn, methods=["GET"]))
