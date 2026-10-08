"""
여러 번 내려받은 HJNC 엑셀 → "시점별 터미널 예정" 이력 + 리드타임별 예정 오차.

HJNC 엑셀 한 장 = 그 시각의 스냅샷이다. 앞으로 올 배의 입항일시·출항일시는 그 시각의 예정이다.
그래서 크롤러 없이도, 하루 몇 번(예: 08·14·20시) 엑셀을 내려받아 data/snapshots/ 에 쌓으면
24·48·72시간 전 예정이 생긴다 (확장 문서 1순위 데이터).

  python scripts/plan_history.py data/snapshots/*.xlsx
  → reports/plan_history.csv          : 모선항차 × 스냅샷 시각별 예정 접안·출항·물량
  → reports/plan_error_by_lead.csv    : 리드타임 구간별 터미널 예정 오차 (가장 최근 엑셀에서 끝난 배 기준)

스냅샷 시각은 파일명의 14자리 숫자(Berth_Schedule_20261008014718.xlsx)에서 읽는다.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.hjnc import read_berth_excel

BINS = [0, 12, 24, 48, 72, 24 * 14]
LABELS = ["0-12h", "12-24h", "24-48h", "48-72h", "72h+"]


def snap_time(path: str) -> pd.Timestamp:
    m = re.search(r"(20\d{12})", os.path.basename(path))
    if not m:
        raise ValueError(f"파일명에 내보낸 시각(14자리)이 없습니다: {path}")
    return pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S")


def main():
    paths = sorted({p for a in (sys.argv[1:] or ["data/snapshots/*.xlsx"]) for p in glob.glob(a)}, key=snap_time)
    if not paths:
        print("엑셀이 없습니다. data/snapshots/ 에 HJNC 엑셀을 모아 주세요.")
        return
    rows = []
    for p in paths:
        t = snap_time(p)
        df = read_berth_excel(p)
        fut = df[df["작업 시작일시"].isna() & (df["입항일시"] > t)]
        for _, r in fut.iterrows():
            rows.append({"call_id": r["모선항차"], "vessel": r["선박명"], "route": r["항로"], "berth": r["선석"],
                         "snapshot": t, "planned_berth": r["입항일시"], "planned_departure": r["출항일시"],
                         "planned_box": r["box"]})
    hist = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    hist.to_csv("reports/plan_history.csv", index=False, encoding="utf-8-sig")

    last = read_berth_excel(paths[-1])
    done = last[last["done"]].drop_duplicates("모선항차", keep="last").set_index("모선항차")
    h = hist[hist["call_id"].isin(done.index)].copy()
    if h.empty:
        print(f"스냅샷 {len(paths)}개, 예정 {len(hist)}건. 아직 끝난 배가 없어 오차를 계산할 수 없습니다.")
        return
    d = done.loc[h["call_id"]]
    h["actual_berth"] = d["입항일시"].values
    h["final_departure"] = d["출항일시"].values
    h["work_complete"] = d["작업 완료일시"].values
    h["lead_h"] = (h["actual_berth"] - h["snapshot"]).dt.total_seconds() / 3600
    h["berth_err_h"] = (h["actual_berth"] - h["planned_berth"]).dt.total_seconds() / 3600
    h["departure_err_h"] = (h["final_departure"] - h["planned_departure"]).dt.total_seconds() / 3600
    h["lead"] = pd.cut(h["lead_h"], BINS, labels=LABELS)
    g = h.groupby("lead", observed=True)
    out = pd.DataFrame({"pairs": g.size(), "vessels": g["call_id"].nunique(),
                        "berth_mae_h": g["berth_err_h"].apply(lambda x: x.abs().mean()),
                        "berth_mean_h": g["berth_err_h"].mean(),
                        "berth_over_2h": g["berth_err_h"].apply(lambda x: (x.abs() > 2).mean()),
                        "departure_mae_h": g["departure_err_h"].apply(lambda x: x.abs().mean())}).round(3)
    out.to_csv("reports/plan_error_by_lead.csv", encoding="utf-8-sig")
    print(f"스냅샷 {len(paths)}개 ({paths[0]} ~ {paths[-1]}), 예정 {len(hist)}건, 끝난 배와 짝지은 것 {len(h)}건")
    print(out.to_string())


if __name__ == "__main__":
    main()
