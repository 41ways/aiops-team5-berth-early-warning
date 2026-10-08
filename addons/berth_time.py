"""
애드온 1: 선석 체류시간 분해 — 배가 선석을 차지한 시간이 어디에 쓰였나.

  선석 체류 = [접안 → 작업 시작 대기] + [작업] + [작업 끝 → 출항 대기]
  - 작업 시작 대기: 배는 붙었는데 크레인·인력이 아직 안 붙은 시간 (터미널 운영 문제)
  - 작업:           물량 ÷ 생산성. 같은 물량인데 길면 생산성 손실
  - 출항 대기:      작업은 끝났는데 못 떠나는 시간 (도선·예선·서류). 출항일시는 확정 출항 시각(정각)이라 대략값
기존 모델 코드는 읽기만 한다 (data.hjnc.read_berth_excel).
"""
import numpy as np
import pandas as pd

BERTHS = ["1B", "2B", "3B", "4B"]


def breakdown(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["done"] & df["작업 시작일시"].notna() & df["선석"].isin(BERTHS) & (df["box"] >= 50)].copy()
    d["dwell_h"] = d["berth_h"]
    d["wait_start_h"] = d["pre_h"].clip(lower=0)
    d["work_h2"] = d["work_h"].clip(lower=0)
    d["wait_depart_h"] = d["post_h"].clip(lower=0)
    d = d[(d["dwell_h"] > 0) & (d["dwell_h"] < 240)]
    d["prod"] = d["box"] / d["work_h2"].replace(0, np.nan)
    d["month"] = d["입항일시"].dt.to_period("M").astype(str)
    # 생산성 손실 시간: 그 달 중앙 생산성으로 했으면 걸렸을 시간보다 더 걸린 시간 (0 이상)
    med = d.groupby("month")["prod"].transform("median")
    d["slow_work_h"] = (d["work_h2"] - d["box"] / med).clip(lower=0)
    d["idle_h"] = d["wait_start_h"] + d["wait_depart_h"]
    return d


def summaries(d: pd.DataFrame) -> dict:
    agg = dict(척수=("box", "size"), 평균체류_h=("dwell_h", "mean"), 작업시작대기_h=("wait_start_h", "mean"),
               작업_h=("work_h2", "mean"), 출항대기_h=("wait_depart_h", "mean"), 생산성_box_h=("prod", "median"),
               생산성손실_h=("slow_work_h", "mean"))
    by_month = d.groupby("month").agg(**agg).round(2)
    by_berth = d.groupby("선석").agg(**agg).round(2)
    by_route = d.groupby("항로").agg(**agg).query("척수 >= 10").sort_values("작업시작대기_h", ascending=False).round(2).head(10)
    worst = d.sort_values("idle_h", ascending=False).head(15)[
        ["선박명", "모선항차", "선석", "항로", "입항일시", "box", "wait_start_h", "work_h2", "wait_depart_h", "idle_h"]].round(2)
    tot = d[["wait_start_h", "work_h2", "wait_depart_h"]].sum()
    share = (tot / tot.sum() * 100).round(1)
    headline = {
        "기간": f"{d['입항일시'].min():%Y-%m-%d} ~ {d['입항일시'].max():%Y-%m-%d}", "척수": int(len(d)),
        "체류 중 작업시작대기 비중(%)": float(share["wait_start_h"]), "체류 중 작업 비중(%)": float(share["work_h2"]),
        "체류 중 출항대기 비중(%)": float(share["wait_depart_h"]),
        "작업시작 1시간 넘게 대기한 배(%)": round(float((d["wait_start_h"] > 1).mean() * 100), 1),
        "출항 2시간 넘게 대기한 배(%)": round(float((d["wait_depart_h"] > 2).mean() * 100), 1),
        "선석이 놀았던 시간 합계(h, 대기 합)": round(float(d["idle_h"].sum()), 0),
        "생산성 손실 시간 합계(h)": round(float(d["slow_work_h"].sum()), 0),
    }
    return {"headline": headline, "by_month": by_month, "by_berth": by_berth, "by_route": by_route, "worst": worst}


def daily_capacity(df: pd.DataFrame, start="2025-01-01", end=None, n_berths: int = 4) -> pd.DataFrame:
    """과거 하루별: 도착 척수, 평균 점유 선석, 4개 선석이 다 찬 시간."""
    d = df[df["done"] & df["선석"].isin(BERTHS)]
    end = end or d["출항일시"].max().normalize()
    hrs = pd.date_range(start, end, freq="h", inclusive="left")
    occ = np.zeros(len(hrs))
    t0 = hrs[0]
    for a, b in zip(d["입항일시"], d["출항일시"]):
        i, j = int((a - t0) / pd.Timedelta(hours=1)), int((b - t0) / pd.Timedelta(hours=1))
        occ[max(i, 0):max(min(j, len(hrs)), 0)] += 1
    o = pd.Series(occ, index=hrs)
    arr = d.set_index("입항일시").resample("D").size().reindex(o.resample("D").mean().index, fill_value=0)
    return pd.DataFrame({"도착척수": arr, "평균점유선석": o.resample("D").mean().round(2),
                         "점유율": (o.resample("D").mean() / n_berths).round(3),
                         "만석시간_h": (o >= n_berths).resample("D").sum()})
