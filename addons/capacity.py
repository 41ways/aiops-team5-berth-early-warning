"""
애드온 2: 도착 몰림·수용 능력 계산기 — 기존 모델의 예측을 받아 앞으로 며칠의 시간대별 부하를 계산한다.

  입력: 기존 예측표와 같은 예측 (scripts/predict_tomorrow.collect 를 그대로 호출, 코드 수정 없음)
  시간대별: 차 있는 선석 수, 작업 중인 배 수, 시간당 처리 박스(box/h), 필요 크레인(QC) 대수
  두 시나리오: 예측대로 / 작업이 늦어질 때(작업시간 × 80% 범위 위쪽 배수, 기본 1.32)
  경고: 선석 부족(동시에 필요한 선석 > 4), 처리 한계 초과(시간당 처리 박스 > 과거 실적 99% 분위), 도착 몰림(6시간 안 3척 이상),
        같은 선석 겹침, 접안 직후 작업 대기 위험(최근 그 선석의 작업시작 대기가 긴 경우)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

N_BERTHS = 4   # 엑셀 선석 번호 수 (물리 선석은 1.1km에 3선석 — addons/rules.py)


from addons.rules import v as rule
MIN_GAP = rule("min_gap_between_vessels_h")


def _band():
    p = os.path.join(ROOT, "serving_app/models/residual_band.json")
    return json.load(open(p)) if os.path.exists(p) else {"work_x90": 1.32}


def vessel_plan(path: str, as_of: pd.Timestamp, days: int = 3) -> pd.DataFrame:
    from predict_tomorrow import collect   # 기존 예측 그대로 (읽기만)
    df, rows, est, end = collect(path, as_of, days)
    x90 = _band().get("work_x90", 1.32)
    out = []
    for x in rows:
        r = x["r"]
        working = x["kind"] == "작업중"
        berth = r["입항일시"] + pd.Timedelta(hours=x["delay_h"])
        ws = r["작업 시작일시"] if working else berth + pd.Timedelta(hours=est.pre_h)
        we = ws + pd.Timedelta(hours=x["work_h"])
        we_late = ws + pd.Timedelta(hours=x["work_h"] * x90)
        out.append({"구분": "작업중" if working else "예정", "선석": r["선석"], "선박명": r["선박명"], "모선항차": r["모선항차"],
                    "항로": r["항로"], "물량": int(r["box"]), "터미널예정접안": r["입항일시"], "예측접안": berth,
                    "예측작업시작": ws, "예측작업완료": we, "예측출항": we + pd.Timedelta(hours=est.post_h),
                    "늦을때_작업완료": we_late, "늦을때_출항": we_late + pd.Timedelta(hours=est.post_h),
                    "80%안전_다음배접안가능": we_late + pd.Timedelta(hours=est.post_h + MIN_GAP),
                    "터미널예정출항": r["출항일시"], "작업시간_h": round(x["work_h"], 1),
                    "시간당처리_box": round(r["box"] / max(x["work_h"], 0.5), 1)})
    return pd.DataFrame(out).sort_values("예측접안").reset_index(drop=True), end


def hourly_load(v: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, qc_rate: float, late: bool = False) -> pd.DataFrame:
    hrs = pd.date_range(start.floor("h"), end, freq="h", inclusive="left")
    dep, we = ("늦을때_출항", "늦을때_작업완료") if late else ("예측출항", "예측작업완료")
    rows = []
    for t in hrs:
        mid = t + pd.Timedelta(minutes=30)
        on_berth = v[(v["예측접안"] <= mid) & (v[dep] > mid)]
        working = v[(v["예측작업시작"] <= mid) & (v[we] > mid)]
        rate = working["물량"] / ((working[we] - working["예측작업시작"]).dt.total_seconds() / 3600).clip(lower=0.5)
        box_h = float(rate.sum())
        rows.append({"시각": t, "차있는선석": len(on_berth), "작업중인배": len(working), "시간당처리_box": round(box_h, 0),
                     "필요QC": int(np.ceil(box_h / qc_rate)) if box_h > 0 else 0})
    return pd.DataFrame(rows)


def historical_capacity(df: pd.DataFrame, q: float = 0.99) -> dict:
    """과거 실적으로 본 처리 한계: 시간당 처리 박스의 99% 분위, 동시에 선석에 있던 배 수의 최대.
    가정값(QC 대수 × 대당 처리량) 대신 '이 터미널이 실제로 해 본 최대치'를 기준으로 경고한다."""
    d = df[df["done"] & df["작업 시작일시"].notna() & (df["box"] >= 50) & df["선석"].isin(["1B", "2B", "3B", "4B"])]
    d = d[d["입항일시"] >= "2025-01-01"]
    hrs = pd.date_range("2025-01-01", d["출항일시"].max().ceil("h"), freq="h")
    t0, occ, br = hrs[0], np.zeros(len(hrs)), np.zeros(len(hrs))
    for a, b, s_, e_, bx, wh in zip(d["입항일시"], d["출항일시"], d["작업 시작일시"], d["작업 완료일시"], d["box"], d["work_h"]):
        i, j = int((a - t0) / pd.Timedelta(hours=1)), int((b - t0) / pd.Timedelta(hours=1))
        occ[max(i, 0):max(min(j, len(hrs)), 0)] += 1
        i, j = int((s_ - t0) / pd.Timedelta(hours=1)), int((e_ - t0) / pd.Timedelta(hours=1))
        br[max(i, 0):max(min(j, len(hrs)), 0)] += bx / max(wh, 0.5)
    return {"box_h_p99": float(np.percentile(br[br > 0], q * 100)), "box_h_p50": float(np.percentile(br[br > 0], 50)),
            "hours_5_on_berth": int((occ >= 5).sum()), "hours_total": int(len(occ))}


def warnings(v: pd.DataFrame, load: pd.DataFrame, load_late: pd.DataFrame, hist: dict, recent_wait: dict) -> pd.DataFrame:
    w = []
    def runs(mask, frame):
        """연속된 시간 구간으로 묶기."""
        out, s = [], None
        for t, m in zip(frame["시각"], mask):
            if m and s is None:
                s = t
            if not m and s is not None:
                out.append((s, t)); s = None
        if s is not None:
            out.append((s, frame["시각"].iloc[-1] + pd.Timedelta(hours=1)))
        return out
    for name, L in (("예측대로", load), ("작업이 늦어질 때", load_late)):
        for s, e in runs(L["차있는선석"] > N_BERTHS, L):
            w.append({"종류": "선석 부족", "시나리오": name, "시작": s, "끝": e,
                      "내용": f"동시에 {int(L[(L['시각'] >= s) & (L['시각'] < e)]['차있는선석'].max())}척이 선석 필요 (선석 번호 {N_BERTHS}개). "
                              f"과거 21개월 동안 5척이 동시에 붙어 있던 시간은 {hist['hours_5_on_berth']}시간뿐 → 드문 상황"})
        cap = hist["box_h_p99"]
        for s, e in runs(L["시간당처리_box"] > cap, L):
            seg = L[(L["시각"] >= s) & (L["시각"] < e)]
            w.append({"종류": "처리 한계 초과", "시나리오": name, "시작": s, "끝": e,
                      "내용": f"시간당 최대 {int(seg['시간당처리_box'].max())} box 필요 > 과거 실적 99% 분위 {cap:.0f} box/h "
                              f"(QC 환산 {int(seg['필요QC'].max())}대) → 크레인·갱 추가 또는 작업 순서 조정 검토"})
    arr = v[v["구분"] == "예정"].sort_values("예측접안")
    for i in range(len(arr)):
        t = arr["예측접안"].iloc[i]
        win = arr[(arr["예측접안"] >= t) & (arr["예측접안"] < t + pd.Timedelta(hours=6))]
        if len(win) >= 3 and not any(x["종류"] == "도착 몰림" and x["시작"] <= t < x["끝"] for x in w):
            w.append({"종류": "도착 몰림", "시나리오": "예측대로", "시작": t, "끝": win["예측접안"].max(),
                      "내용": f"6시간 안 {len(win)}척 접안: " + ", ".join(win["선박명"])})
    for b, g in v.groupby("선석"):
        g = g.sort_values("예측접안")
        for (_, p), (_, c) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            gap = (c["예측접안"] - p["예측출항"]).total_seconds() / 3600
            gap_late = (c["예측접안"] - p["늦을때_출항"]).total_seconds() / 3600
            if gap < 0 or gap_late < 0:
                w.append({"종류": "같은 선석 겹침", "시나리오": "예측대로" if gap < 0 else "작업이 늦어질 때",
                          "시작": c["예측접안"], "끝": p["늦을때_출항"] if gap >= 0 else p["예측출항"],
                          "내용": f"{b}: {p['선박명']} 출항 전 {c['선박명']} 접안 (간격 {gap:+.1f}h, 늦으면 {gap_late:+.1f}h). "
                                  "같은 선석 번호에서 안벽을 나눠 쓰는지 확인"})
    for _, r in v[v["구분"] == "예정"].iterrows():
        rw = recent_wait.get(r["선석"])
        if rw is not None and rw > 1.0:
            w.append({"종류": "접안 후 작업 대기 위험", "시나리오": "최근 실적", "시작": r["예측접안"], "끝": r["예측작업시작"],
                      "내용": f"{r['선석']} 최근 30척 평균 작업시작 대기 {rw:.1f}h → {r['선박명']} 크레인·갱 배치 미리 확인"})
    return pd.DataFrame(w).sort_values(["시작", "종류"]).reset_index(drop=True) if w else pd.DataFrame(
        columns=["종류", "시나리오", "시작", "끝", "내용"])
