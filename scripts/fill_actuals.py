"""
예측표의 노란 칸(실제 접안·실제 출항)을 새로 받은 HJNC 엑셀로 자동 채우기.

  python scripts/fill_actuals.py 오늘_접안예측표_v1.0.xlsx Berth_Schedule_새로받은것.xlsx [--out 결과.xlsx]

  - 새 엑셀에서 '작업 시작일시'가 있는 배 = 접안 완료 → 입항일시가 실제 접안
  - '작업 완료일시'가 있는 배 = 작업 끝 → 출항일시(터미널 확정 출항, 정각 단위)와 작업완료일시(실측)
  - 모선항차(E열)로 맞춘다. 이미 손으로 넣은 값은 덮어쓰지 않는다 (--overwrite 로 덮어쓰기)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from openpyxl import load_workbook

from data.hjnc import read_berth_excel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_xlsx")
    ap.add_argument("new_export")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    df = read_berth_excel(a.new_export).drop_duplicates("모선항차", keep="last").set_index("모선항차")
    wb = load_workbook(a.pred_xlsx)
    ws = wb["예측표"]
    n_b = n_d = 0
    for row in range(6, ws.max_row + 1):
        cid = ws.cell(row, 5).value
        if not cid or cid not in df.index:
            continue
        r = df.loc[cid]
        if pd.notna(r["작업 시작일시"]) and ws.cell(row, 2).value != "작업중" and (a.overwrite or ws.cell(row, 13).value is None):
            ws.cell(row, 13).value = r["입항일시"].to_pydatetime()
            n_b += 1
        if pd.notna(r["작업 완료일시"]) and (a.overwrite or ws.cell(row, 22).value is None):
            ws.cell(row, 22).value = r["출항일시"].to_pydatetime()
            n_d += 1
        if pd.notna(r["작업 완료일시"]) and (a.overwrite or ws.cell(row, 26).value is None):
            ws.cell(row, 26).value = r["작업 완료일시"].to_pydatetime()
    wb.calculation.fullCalcOnLoad = True
    out = a.out or a.pred_xlsx.replace(".xlsx", "_실적입력.xlsx")
    wb.save(out)
    print(f"{out}: 실제 접안 {n_b}척, 실제 출항 {n_d}척 채움. 엑셀에서 열면 '정확도' 시트가 다시 계산됩니다.")


if __name__ == "__main__":
    main()
