"""
v2.5 성능 점검용 별도 서버: "오늘이 그날이라고 치고" HJNC 실데이터로 예측 → 실제 접안·출항과 비교.

  - 데이터: data/real/hjnc_berth_schedule_20261008014718.csv 고정 (업로드 없음)
  - 기준 시각 t0 (기본 그날 00:00) 에 알 수 있는 것만 쓴다
      접안 예측 입력 : t0 까지 접안한 배의 이력 (같은 항로 최근 20척)
      출항 예측 입력 : t0 전에 작업이 끝난 배로 맞춘 작업시간 추정 (WorkTimeEstimator, fit_before=t0)
  - 대상 배: 터미널 예정 접안(반입 마감 + 10h)이 [t0, t0 + hours) 이고 t0에 아직 접안 전인 배
  - 정답: 실제 접안 = 입항일시, 출항 = 출항일시(확정 출항, 정각이 많음)와 작업 완료일시 둘 다 보여 준다
  - 드리프트·재학습·로그 없음. 모델은 로컬 serving_app/models/arrival_v1.keras (HJNC 2025 학습)

실행 (app/ 에서):
  uvicorn addons.backtest_server:app --host 0.0.0.0 --port 8090   → http://localhost:8090/
"""
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from data.features import GROUP_KEY, TRAIN_END, VAL_END, _history_for, berthed_history, sequence_points
from data.hjnc import read_berth_excel
from serving_app import model_loader
from serving_app.schedule import WorkTimeEstimator

DATA = "data/real/hjnc_berth_schedule_20261008014718.csv"
HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_server.html")

app = FastAPI(title="HJNC 접안 예측 성능 점검 (v2.5)")
_c: dict = {}


def _df():
    if "df" not in _c:
        _c["df"] = read_berth_excel(DATA)
        _c["hist"] = berthed_history(_c["df"])
    return _c["df"], _c["hist"]


def _f(t):
    return None if pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def _h(a, b):
    return None if pd.isna(a) or pd.isna(b) else round((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 3600, 3)


@app.get("/")
def page():
    return FileResponse(HTML, media_type="text/html")


@app.get("/api")
def api(as_of: str = "2026-10-05 00:00", hours: float = 24.0):
    try:
        t0 = pd.Timestamp(as_of)
    except ValueError:
        raise HTTPException(422, "as_of 형식은 'YYYY-MM-DD HH:MM'")
    df, hist = _df()
    plan = df["plan_berth_10h"]
    win = df[(plan >= t0) & (plan < t0 + pd.Timedelta(hours=hours))]
    already = win[win["작업 시작일시"].notna() & (win["입항일시"] <= t0)]
    target = win.drop(already.index)

    model = model_loader.get_model()
    est = WorkTimeEstimator(df, fit_before=t0)
    seqs, rows, skipped = [], [], []
    for _, r in target.iterrows():
        h = _history_for(hist, r[GROUP_KEY], t0)
        if h is None:
            skipped.append(r["선박명"])
            continue
        seqs.append(sequence_points(h))
        rows.append(r)
    preds = model.predict_many(seqs) if seqs else []

    out = []
    for r, d in zip(rows, preds):
        p = r["plan_berth_10h"]
        pb = p + pd.Timedelta(hours=float(d))
        berthed = pd.notna(r["작업 시작일시"])
        act = r["입항일시"] if berthed else pd.NaT
        pdep = est.departure(pb, r["box"], r[GROUP_KEY], t0)
        pdone = pb + pd.Timedelta(hours=est.pre_h + est.work_hours(r["box"], r[GROUP_KEY], t0))
        done = pd.notna(r["작업 완료일시"])
        out.append({
            "vessel": r["선박명"], "call_id": r["모선항차"], "berth_no": r["선석"], "route": r[GROUP_KEY], "line": r["선사"],
            "box": int(r["box"]), "cct": _f(r["반입 마감일시"]),
            "plan_berth": _f(p), "pred_berth": _f(pb), "pred_delay_h": round(float(d), 3),
            "actual_berth": _f(act), "actual_delay_h": _h(p, act),
            "term_err_h": _h(p, act), "model_err_h": _h(pb, act),
            "pred_departure": _f(pdep), "actual_departure": _f(r["출항일시"]) if done else None,
            "dep_err_h": _h(pdep, r["출항일시"]) if done else None,
            "pred_work_done": _f(pdone), "actual_work_done": _f(r["작업 완료일시"]),
            "work_done_err_h": _h(pdone, r["작업 완료일시"]) if done else None,
            "status": "완료" if done else ("작업 중" if berthed else "미접안(데이터 시점 이후)")})
    out.sort(key=lambda x: x["plan_berth"])

    ev = [x for x in out if x["actual_berth"] and abs(x["term_err_h"]) <= 48]
    dv = [x for x in out if x["dep_err_h"] is not None]

    def mae(k, xs):
        return round(sum(abs(x[k]) for x in xs) / len(xs), 3) if xs else None

    def share(k, xs, lim):
        return round(sum(abs(x[k]) <= lim for x in xs) / len(xs), 3) if xs else None

    phase = "학습 구간(2025) — 모델이 이미 본 배라 성적이 좋게 나옴" if t0 < TRAIN_END else \
        "검증 구간(2026-01~04) — 게이트 판단에 쓴 기간" if t0 < VAL_END else "테스트 구간(2026-05~) — 모델이 본 적 없는 배"
    return {"as_of": _f(t0), "hours": hours, "phase": phase, "model_version": model.version,
            "data_file": os.path.basename(DATA), "data_last_berth": _f(df["작업 시작일시"].max()),
            "vessels": out, "already_berthed_before_as_of": already["선박명"].tolist(), "skipped_no_history": skipped,
            "summary": {"n": len(out), "n_eval": len(ev),
                        "term_mae_h": mae("term_err_h", ev), "model_mae_h": mae("model_err_h", ev),
                        "term_in30": share("term_err_h", ev, 0.5), "model_in30": share("model_err_h", ev, 0.5),
                        "term_over1h": sum(abs(x["term_err_h"]) > 1 for x in ev), "model_over1h": sum(abs(x["model_err_h"]) > 1 for x in ev),
                        "model_closer": sum(abs(x["model_err_h"]) < abs(x["term_err_h"]) for x in ev),
                        "term_closer": sum(abs(x["model_err_h"]) > abs(x["term_err_h"]) for x in ev),
                        "n_dep": len(dv), "dep_mae_h": mae("dep_err_h", dv), "dep_in3h": share("dep_err_h", dv, 3.0),
                        "work_done_mae_h": mae("work_done_err_h", dv)}}
