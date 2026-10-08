"""
v2.5: 배 이름 → 기본 제원 표 (부산항만공사 선박 입출항 집계 2019~2024 CSV).

터미널 선석 스케줄에는 총톤수·국적·앞 항구가 없어서, 공개 CSV에서 같은 이름의 배를 찾아 붙인다.
CSV는 58MB라 저장소에 넣지 않고, 여기서 만든 작은 JSON(addons/vessel_info.json)만 쓴다.
2024년 이후 처음 온 배나 이름이 바뀐 배는 비어 있다 (대시보드에 "정보 없음").

실행 (app/ 에서):
  python addons/vessel_info.py <BPA CSV 경로>
  예) python addons/vessel_info.py ../../../port-arrival-inputs/downloads/20261008_bpa_선박입출항집계_2019-2024.csv
출처: https://www.data.go.kr/data/15150637/fileData.do
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.hjnc import read_berth_excel

OUT = "addons/vessel_info.json"
HJNC = "data/real/sinhang5_berth_schedule.csv"  # v2.5: 신항 5개 터미널 통합 파일의 배 이름


def build(bpa_csv: str, hjnc_path: str = HJNC) -> dict:
    names = set(read_berth_excel(hjnc_path)["선박명"].astype(str).str.strip().str.upper())
    b = pd.read_csv(bpa_csv, encoding="cp949", low_memory=False)
    b["nm"] = b["선박명"].astype(str).str.strip().str.upper()
    b = b[b["nm"].isin(names)].copy()
    b["입항일시"] = pd.to_datetime(b["입항일시"], errors="coerce")
    b = b[b["입항일시"] >= "2019-01-01"].sort_values("입항일시")
    out = {}
    for nm, g in b.groupby("nm"):
        last = g.iloc[-1]
        gt = last["국제총톤수"] if pd.notna(last["국제총톤수"]) and last["국제총톤수"] > 0 else last["총톤수"]
        prev = g["전출항지명"].dropna().astype(str)
        out[nm] = {
            "gt": int(gt) if pd.notna(gt) else None,
            "flag": str(last["선박국적명"]) if pd.notna(last["선박국적명"]) else None,
            "type": str(last["선박종류명"]) if pd.notna(last["선박종류명"]) else None,
            "prev_port_last": str(last["전출항지명"]) if pd.notna(last["전출항지명"]) else None,
            "prev_port_top": prev.value_counts().index[0] if len(prev) else None,
            "next_port_last": str(last["차항지명"]) if pd.notna(last["차항지명"]) else None,
            "crew": int(g["선원수"].median()) if g["선원수"].notna().any() else None,
            "calls_2019_2024": int(len(g)),
            "last_seen": last["입항일시"].strftime("%Y-%m-%d"),
        }
    return {"source": "부산항만공사_선박 입출항 집계 정보 2019~2024 (data.go.kr 15150637)",
            "matched": len(out), "hjnc_vessels": len(names), "vessels": out}


if __name__ == "__main__":
    info = build(sys.argv[1])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(f"{OUT}: HJNC 배 {info['hjnc_vessels']}척 중 {info['matched']}척 제원 찾음")
