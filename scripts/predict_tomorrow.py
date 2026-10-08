"""
오늘·내일 접안 예측표 + 선석 일정표 + 시간대별 작업량 엑셀 만들기.

  python scripts/predict_tomorrow.py [HJNC 엑셀] [--as-of "2026-10-08 01:47"] [--days 3] [--out 파일명.xlsx]

  - 예측 대상: as_of 이후 접안 예정인 배 (as_of 날짜부터 days일), + 지금 작업 중인 배(출항 예측만)
  - 접안: 모델(LSTM, 같은 항로 최근 20척) 보정값을 터미널 예정에 더함
  - 출항: 예측 접안 + 접안→작업시작 중앙값 + 예측 작업시간(물량·항로) + 작업완료→출항 중앙값
  - 노란 칸(실제 접안·실제 출항)에 실제 시각을 넣으면 오차·정확도가 자동 계산된다.
    새로 받은 HJNC 엑셀로 한 번에 채우려면 scripts/fill_actuals.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from tensorflow import keras

from data.features import ArrivalScaler, berthed_history, _history_for, sequence_points
from data.hjnc import read_berth_excel
from data.storage import latest_upload
from serving_app.schedule import WorkTimeEstimator

MODEL_PATH = "serving_app/models/arrival_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
BERTHS = ["1B", "2B", "3B", "4B"]

F = "Arial"
BOLD = Font(name=F, bold=True)
NORMAL = Font(name=F)
BLUE = Font(name=F, color="0000FF")
TITLE = Font(name=F, bold=True, size=14)
SMALL = Font(name=F, size=9, color="555555")
HEAD_FILL = PatternFill("solid", fgColor="DDE5F0")
INPUT_FILL = PatternFill("solid", fgColor="FFFF00")
DONE_FILL = PatternFill("solid", fgColor="EEEEEE")
PLAN_FILL = PatternFill("solid", fgColor="BFD7EA")
MODEL_FILL = PatternFill("solid", fgColor="F6C28B")
WARN_FILL = PatternFill("solid", fgColor="F4B6B6")
THIN = Side(style="thin", color="BBBBBB")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
DT = "yyyy-mm-dd hh:mm"
HM = "mm/dd hh:mm"


def collect(path: str, as_of: pd.Timestamp, days: int):
    df = read_berth_excel(path)
    hist = berthed_history(df)
    end = as_of.normalize() + pd.Timedelta(days=days)
    working = df[(~df["done"]) & df["작업 시작일시"].notna() & (df["작업 시작일시"] > as_of - pd.Timedelta(days=7))
                 & (df["작업 시작일시"] <= as_of)].sort_values("입항일시")
    future = df[(~df["done"]) & df["작업 시작일시"].isna() & (df["입항일시"] > as_of) & (df["입항일시"] < end)
                ].sort_values("입항일시")

    scaler = ArrivalScaler.load(SCALER_PATH)
    model = keras.models.load_model(MODEL_PATH)
    rows = []
    for _, r in working.iterrows():
        rows.append({"kind": "작업중", "r": r, "delay_h": 0.0})
    seqs, idx = [], []
    for _, r in future.iterrows():
        h = _history_for(hist, r["항로"], as_of)
        rows.append({"kind": "예정", "r": r, "delay_h": 0.0})
        if h is not None:
            seqs.append([scaler.transform_point(p["delay_hours"], p["box"]) for p in sequence_points(h)])
            idx.append(len(rows) - 1)
    if seqs:
        import numpy as np
        preds = model.predict(np.array(seqs, dtype="float32"), verbose=0).flatten()
        for i, p in zip(idx, preds):
            rows[i]["delay_h"] = scaler.inverse_target(float(p))
    est = WorkTimeEstimator(df)
    for x in rows:
        x["work_h"] = est.work_hours(x["r"]["box"], x["r"]["항로"], as_of)
    return df, rows, est, end


def day_label(t: pd.Timestamp, as_of: pd.Timestamp) -> str:
    d = (t.normalize() - as_of.normalize()).days
    return {0: "오늘", 1: "내일", 2: "모레"}.get(d, f"+{d}일") + f" {t:%m/%d}"


def load_band() -> dict:
    import json
    p = "serving_app/models/residual_band.json"
    return json.load(open(p)) if os.path.exists(p) else {"q10": -0.31, "q90": 0.33}


def build(path, as_of, days, out):
    df, rows, est, end = collect(path, as_of, days)
    band = load_band()
    wb = Workbook()

    # ---------------- 가정 ----------------
    ga = wb.active
    ga.title = "가정"
    ga["A1"], ga["A1"].font = "가정·설명", TITLE
    assumptions = [
        ("접안→작업시작 (시간)", round(est.pre_h, 2), "HJNC 완료 선박 중앙값 (같은 엑셀에서 계산)"),
        ("작업완료→출항 (시간)", round(est.post_h, 2), "HJNC 완료 선박 중앙값 (같은 엑셀에서 계산)"),
        ("크레인(QC) 1기당 처리량 (box/시간)", 25, "가정값. 부산 신항 QC 순작업 생산성 대략치 — 터미널 실제값으로 바꿔 쓰세요"),
        ("QC 1기당 작업 인원 (명/교대)", 8, "가정값. 갱(gang) 구성 인원 — 터미널 실제값으로 바꿔 쓰세요"),
        ("같은 선석 간격 주의 기준 (시간)", 2, "직전 배 예측 출항과 다음 배 예측 접안 사이가 이보다 짧으면 표시"),
        ("예측 범위 아래쪽 (시간, 10%)", band["q10"], "검증 구간(2026-01~04) 모델 오차의 10% 분위. 테스트에서 80% 범위가 78%를 덮었다"),
        ("예측 범위 위쪽 (시간, 90%)", band["q90"], "검증 구간 모델 오차의 90% 분위 (serving_app/models/residual_band.json)"),
        ("작업시간 범위 아래 배수 (10%)", band.get("work_x10", 0.78), "검증 구간 실제 ÷ 예측 작업시간의 10% 분위. 테스트에서 80% 범위가 79%를 덮었다"),
        ("작업시간 범위 위 배수 (90%)", band.get("work_x90", 1.32), "검증 구간 실제 ÷ 예측 작업시간의 90% 분위"),
    ]
    ga["A3"], ga["B3"], ga["C3"] = "항목", "값", "근거"
    for c in "ABC":
        ga[f"{c}3"].font, ga[f"{c}3"].fill = BOLD, HEAD_FILL
    for i, (k, v, s) in enumerate(assumptions, start=4):
        ga[f"A{i}"], ga[f"B{i}"], ga[f"C{i}"] = k, v, s
        ga[f"A{i}"].font, ga[f"B{i}"].font, ga[f"C{i}"].font = NORMAL, BLUE, SMALL
    notes = [
        f"기준 시각(as_of): {as_of:%Y-%m-%d %H:%M}  /  데이터: {os.path.basename(path)} (HJNC 정보서비스 선석배정현황 엑셀)",
        "모델: BERTH_ARRIVAL_Corrector v1 (LSTM, 같은 항로 최근 20척의 [접안 오차, 물량] → 다음 배 접안 오차)",
        "터미널 예정 접안 = 엑셀의 입항일시(미래 배). 반입 마감 = 예정 접안 − 10시간이며 이 시점이 예측 시점이다.",
        "백테스트(2026-05-01~10-06, 430척): 접안 MAE 터미널 16.5분 → 모델 12.8분, 1시간 넘게 틀린 배 2.3% → 0.9%",
        "출항(선석 반납) 예측 MAE 약 3.3시간 — 크레인 배정 정보가 없어 작업시간 오차가 크다. 참고용.",
        "엑셀의 출항일시는 작업이 끝난 배도 96%가 정각이다 → 실측이 아니라 터미널이 확정한 출항 시각. 실측은 작업완료일시.",
        "같은 선석 겹침: 과거 완료 선박의 3.6%는 같은 선석 번호에서 앞 배와 겹쳤다(안벽 공유). 경고는 '확인 필요' 뜻.",
        "정확도(백테스트)는 반입 마감 시점(접안 10시간 전) 예측 기준이다. 그보다 이른 예측은 '확인 사항'에 표시했고 아직 검증 전이다.",
        "같은 입력(같은 항로 최근 20척)의 단순 평균 보정도 0.218h로 모델(0.213h)과 비슷하다. 개선의 대부분은 입력 설계에서 나온다.",
        "파란 글씨 = 바꿀 수 있는 가정값. 노란 칸 = 실제 시각 입력 칸 (예: 2026-10-08 04:12).",
    ]
    ga["A13"], ga["B13"] = "기준 시각 (as_of)", as_of.to_pydatetime()
    ga["B13"].number_format = DT
    ga["A13"].font = ga["B13"].font = NORMAL
    ga["C13"], ga["C13"].font = "작업 중인 배의 출항 범위가 이 시각보다 앞서지 않게 하는 데 씀", SMALL
    for i, n in enumerate(notes, start=15):
        ga[f"A{i}"], ga[f"A{i}"].font = n, NORMAL
    ga.column_dimensions["A"].width = 38
    ga.column_dimensions["B"].width = 10
    ga.column_dimensions["C"].width = 70

    # ---------------- 예측표 ----------------
    ws = wb.create_sheet("예측표", 0)
    ws["A1"], ws["A1"].font = f"HJNC 접안·출항 예측표  (기준 {as_of:%Y-%m-%d %H:%M}, ~{(end - pd.Timedelta(minutes=1)):%m/%d})", TITLE
    ws["A2"] = "노란 칸에 실제 시각을 넣으면 오차(분)가 계산되고 '정확도' 시트에 모입니다. 입력 예: 2026-10-08 04:12"
    ws["A2"].font = SMALL
    ws["A3"] = "모델 보정(분) = 모델이 터미널 예정에 더한 값.  예측 출항 = 예측 접안 + 접안→작업시작 + 예측 작업시간 + 작업완료→출항 ('가정' 시트)"
    ws["A3"].font = SMALL
    head = ["No", "구분", "선석", "선박명", "모선항차", "선사", "항로", "물량(box)", "반입 마감",
            "터미널 예정 접안", "모델 예측 접안", "모델 보정(분)", "실제 접안 (입력)", "터미널 접안 오차(분)",
            "모델 접안 오차(분)", "더 가까운 쪽(접안)", "터미널 예정 출항", "예측 작업시간(h)", "예측 작업시작",
            "예측 작업완료", "모델 예측 출항", "실제 출항 (입력)", "터미널 출항 오차(분)", "모델 출항 오차(분)",
            "더 가까운 쪽(출항)", "실제 작업완료 (입력)", "모델 작업완료 오차(분)", "모델 접안 80% 범위", "모델 출항 80% 범위", "확인 사항"]
    H = 5
    for j, h in enumerate(head, start=1):
        c = ws.cell(H, j, h)
        c.font, c.fill, c.border = BOLD, HEAD_FILL, BOX
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[H].height = 32

    # 같은 선석 확인 (모델 기준): 직전 배 예측 출항 vs 이 배 예측 접안
    for x in rows:
        r = x["r"]
        x["berth_t"] = r["입항일시"] + pd.Timedelta(hours=x["delay_h"])
        start = r["작업 시작일시"] if x["kind"] == "작업중" else x["berth_t"] + pd.Timedelta(hours=est.pre_h)
        x["dep_t"] = start + pd.Timedelta(hours=x["work_h"] + est.post_h)
    for b in BERTHS + ["OT"]:
        seq = sorted([x for x in rows if x["r"]["선석"] == b], key=lambda x: x["berth_t"])
        for prev, cur in zip(seq, seq[1:]):
            gap = (cur["berth_t"] - prev["dep_t"]).total_seconds() / 3600
            plan_gap = (cur["r"]["입항일시"] - prev["r"]["출항일시"]).total_seconds() / 3600
            if gap < 0:
                cur["flag"] = f"겹침 {-gap:.1f}h: {prev['r']['선박명']} 예측 출항 {prev['dep_t']:%m/%d %H:%M} (터미널 예정 간격 {plan_gap:+.1f}h)"
            elif gap < 2:
                cur["flag"] = f"간격 {gap:.1f}h: {prev['r']['선박명']} 뒤 바로 접안"

    for x in rows:  # 백테스트는 반입 마감(접안 10시간 전) 시점 예측만 검증했다
        cut = x["r"]["반입 마감일시"]
        if x["kind"] != "작업중" and pd.notna(cut) and cut > as_of:
            x["lead_note"] = (f"반입 마감({cut:%m/%d %H:%M}) 전 예측 — 정확도 미검증 구간. "
                              f"마감 시각 이후 새 엑셀로 다시 실행 권장")
    first = H + 1
    for i, x in enumerate(rows):
        r, row = x["r"], first + i
        working = x["kind"] == "작업중"
        v = {1: i + 1, 2: "작업중" if working else day_label(r["입항일시"], as_of), 3: r["선석"], 4: r["선박명"],
             5: r["모선항차"], 6: r["선사"], 7: r["항로"], 8: int(r["box"]),
             9: r["반입 마감일시"].to_pydatetime(), 10: r["입항일시"].to_pydatetime(),
             11: f"=J{row}+L{row}/1440", 12: round(x["delay_h"] * 60, 1),
             14: f'=IF(M{row}="","",ROUND((M{row}-J{row})*1440,0))',
             15: f'=IF(M{row}="","",ROUND((M{row}-K{row})*1440,0))',
             16: f'=IF(M{row}="","",IF(ABS(O{row})<ABS(N{row}),"모델",IF(ABS(O{row})>ABS(N{row}),"터미널","같음")))',
             17: r["출항일시"].to_pydatetime(), 18: round(x["work_h"], 1),
             19: r["작업 시작일시"].to_pydatetime() if working else f"=K{row}+가정!$B$4/24",
             20: f"=S{row}+R{row}/24", 21: f"=T{row}+가정!$B$5/24",
             23: f'=IF(V{row}="","",ROUND((V{row}-Q{row})*1440,0))',
             24: f'=IF(V{row}="","",ROUND((V{row}-U{row})*1440,0))',
             25: f'=IF(V{row}="","",IF(ABS(X{row})<ABS(W{row}),"모델",IF(ABS(X{row})>ABS(W{row}),"터미널","같음")))',
             27: f'=IF(Z{row}="","",ROUND((Z{row}-T{row})*1440,0))',
             28: f'=TEXT(K{row}+가정!$B$9/24,"mm/dd hh:mm")&" ~ "&TEXT(K{row}+가정!$B$10/24,"hh:mm")',
             29: f'=TEXT(MAX(S{row}+R{row}*가정!$B$11/24,가정!$B$13)+가정!$B$5/24,"mm/dd hh:mm")&" ~ "&TEXT(S{row}+R{row}*가정!$B$12/24+가정!$B$5/24,"mm/dd hh:mm")',
             30: " / ".join(t for t in (x.get("flag", ""), x.get("lead_note", "")) if t)}
        if working:  # 이미 접안: 실제 접안을 미리 채우고 접안 오차는 비움 (정확도 집계에서도 제외)
            v[13] = r["입항일시"].to_pydatetime()
            v[14] = v[15] = v[16] = v[28] = None
        for j in range(1, len(head) + 1):
            c = ws.cell(row, j, v.get(j))
            c.font, c.border = NORMAL, BOX
            if j in (9, 10, 11, 13, 17, 19, 20, 21, 22, 26):
                c.number_format = DT
        for j in (13, 22, 26):
            if not (working and j == 13):
                ws.cell(row, j).fill = INPUT_FILL
        ws.cell(row, 12).font = BLUE if not working else NORMAL
        if working:
            for j in range(1, 9):
                ws.cell(row, j).fill = DONE_FILL
        if x.get("flag", "").startswith("겹침"):
            ws.cell(row, 30).fill = WARN_FILL
    last = first + len(rows) - 1
    widths = [4, 10, 5, 18, 16, 6, 6, 8, 16, 16, 16, 8, 16, 9, 9, 9, 16, 8, 16, 16, 16, 16, 9, 9, 9, 16, 9, 20, 26, 70]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(first, 5)
    ws.cell(H, 12).comment = Comment("모델이 예측한 접안 오차(실제−예정)를 분으로. +면 늦게, −면 일찍 접안 예상.", "model")
    ws.cell(H, 22).comment = Comment("HJNC 엑셀의 출항일시. 작업이 끝난 배도 정각 단위(96%)라 측정값이 아니라 "
                                     "터미널이 마지막으로 확정한 출항 시각으로 보인다. 측정값에 가장 가까운 것은 작업완료일시.", "model")
    ws.cell(H, 26).comment = Comment("HJNC 엑셀의 작업 완료일시 (분 단위 실측). 출항보다 믿을 만한 정답.", "model")
    ws.cell(H, 28).comment = Comment("모델 예측 접안 + 검증 구간 오차의 10%~90% 분위. 10척 중 8척이 이 안에 들어오는 게 정상.", "model")
    ws.cell(H, 29).comment = Comment("예측 작업시작 + 예측 작업시간 × (0.78 ~ 1.32) + 작업완료→출항. 10척 중 8척이 이 안에 드는 게 정상.", "model")
    ws.cell(H, 30).comment = Comment("1) 모델 예측 기준으로 같은 선석 번호의 직전 배와 겹치거나 2시간 안에 붙는 경우. "
                                     "2) 반입 마감 전에 한 예측: 백테스트(12.8분)는 마감 시점 예측만 검증했다.", "model")

    # ---------------- 정확도 ----------------
    ac = wb.create_sheet("정확도", 1)
    ac["A1"], ac["A1"].font = "실제 시각 입력 후 정확도 (작업중 배 제외)", TITLE
    rng = lambda col: f"예측표!${col}${first}:${col}${last}"
    sel = f'({rng("B")}<>"작업중")'
    lines = [
        ("접안", None, None),
        ("실제 접안을 입력한 배 (척)", f'=SUMPRODUCT({sel}*({rng("M")}<>""))', "0"),
        ("터미널 예정 평균 오차 (분)", f'=IF(B4=0,"-",SUMPRODUCT({sel}*({rng("M")}<>"")*ABS({rng("M")}-{rng("J")}))*1440/B4)', "0.0"),
        ("모델 예측 평균 오차 (분)", f'=IF(B4=0,"-",SUMPRODUCT({sel}*({rng("M")}<>"")*ABS({rng("M")}-{rng("K")}))*1440/B4)', "0.0"),
        ("모델이 더 가까웠던 배 (척)", f'=COUNTIF({rng("P")},"모델")', "0"),
        ("터미널이 더 가까웠던 배 (척)", f'=COUNTIF({rng("P")},"터미널")', "0"),
        ("모델 80% 범위 안에 든 배 (척)", f'=SUMPRODUCT({sel}*({rng("M")}<>"")*({rng("M")}>={rng("K")}+가정!$B$9/24)*({rng("M")}<={rng("K")}+가정!$B$10/24))', "0"),
        ("출항", None, None),
        ("실제 출항을 입력한 배 (척)", f'=SUMPRODUCT(({rng("V")}<>"")*1)', "0"),
        ("터미널 예정 평균 오차 (분)", f'=IF(B11=0,"-",SUMPRODUCT(({rng("V")}<>"")*ABS({rng("V")}-{rng("Q")}))*1440/B11)', "0.0"),
        ("모델 예측 평균 오차 (분)", f'=IF(B11=0,"-",SUMPRODUCT(({rng("V")}<>"")*ABS({rng("V")}-{rng("U")}))*1440/B11)', "0.0"),
        ("모델이 더 가까웠던 배 (척)", f'=COUNTIF({rng("Y")},"모델")', "0"),
        ("터미널이 더 가까웠던 배 (척)", f'=COUNTIF({rng("Y")},"터미널")', "0"),
        ("작업완료 (터미널은 예정을 공개하지 않음)", None, None),
        ("실제 작업완료를 입력한 배 (척)", f'=SUMPRODUCT(({rng("Z")}<>"")*1)', "0"),
        ("모델 예측 평균 오차 (분)", f'=IF(B17=0,"-",SUMPRODUCT(({rng("Z")}<>"")*ABS({rng("Z")}-{rng("T")}))*1440/B17)', "0.0"),
        ("모델 작업시간 80% 범위 안에 든 배 (척)", f'=SUMPRODUCT(({rng("Z")}<>"")*({rng("Z")}>={rng("S")}+{rng("R")}*가정!$B$11/24)*({rng("Z")}<={rng("S")}+{rng("R")}*가정!$B$12/24))', "0"),
    ]
    for i, (k, f, fmt) in enumerate(lines, start=3):
        ac[f"A{i}"] = k
        ac[f"A{i}"].font = BOLD if f is None else NORMAL
        if f is None:
            ac[f"A{i}"].fill = HEAD_FILL
            continue
        ac[f"B{i}"] = f
        ac[f"B{i}"].number_format = fmt
        ac[f"B{i}"].font = NORMAL
    ac["A21"], ac["A21"].font = "참고: 백테스트 (2026-05-01 ~ 10-06, 430척, reports/backtest_v1.0/summary.json)", BOLD
    ref = [("접안 MAE 터미널 예정", "16.5분"), ("접안 MAE 모델", "12.8분"), ("30분 안에 맞춘 비율 터미널 → 모델", "84.7% → 92.6%"),
           ("1시간 넘게 틀린 배 터미널 → 모델", "2.3% → 0.9%"), ("모델이 터미널보다 나은 날", "155일 중 110일"),
           ("작업완료 시각 MAE 모델(물량·항로 작업시간)", "3.3시간"), ("출항(확정 출항 시각) MAE 모델", "3.3시간"),
           ("같은 입력의 단순 평균 보정 (접안 MAE)", "13.1분 — LSTM(12.8분)과 거의 같음")]
    for i, (k, v) in enumerate(ref, start=22):
        ac[f"A{i}"], ac[f"B{i}"] = k, v
        ac[f"A{i}"].font = ac[f"B{i}"].font = NORMAL
    ac["A30"] = "하루 3척 안팎이라 하루치 결과는 우연에 크게 흔들립니다. 1~2주 모아서 보세요."
    ac["A30"].font = SMALL
    ac["A31"] = "출항일시는 작업이 끝난 배도 정각 단위라 터미널이 확정한 시각으로 보입니다. 측정값은 작업완료일시가 더 정확합니다."
    ac["A31"].font = SMALL
    ac.column_dimensions["A"].width = 40
    ac.column_dimensions["B"].width = 18

    # ---------------- 선석 일정표 (간트) ----------------
    gs = wb.create_sheet("선석일정표", 2)
    t_start = as_of.floor("h")
    t_end = end
    hours = pd.date_range(t_start, t_end - pd.Timedelta(hours=1), freq="h")
    gs["A1"], gs["A1"].font = "선석 일정표 (1칸 = 1시간). 파랑 = 터미널 예정, 주황 = 모델 예측, 빨강 = 모델 예측상 겹침", TITLE
    gs["A2"] = "스크립트 실행 시점의 값으로 칠한 그림입니다 (예측표를 고쳐도 자동으로 바뀌지 않음)."
    gs["A2"].font = SMALL
    gs.cell(4, 1, "선석").font = BOLD
    gs.cell(4, 2, "기준").font = BOLD
    for k, t in enumerate(hours):
        c = gs.cell(3, 3 + k, f"{t:%m/%d}" if (k == 0 or t.hour == 0) else None)
        c.font = SMALL
        c = gs.cell(4, 3 + k, t.hour)
        c.font, c.alignment = SMALL, Alignment(horizontal="center")
        gs.column_dimensions[get_column_letter(3 + k)].width = 3.2
    gs.column_dimensions["A"].width = 6
    gs.column_dimensions["B"].width = 10

    painted = set()

    def paint(rowi, a, b, label, fill):
        cols = [k for k, t in enumerate(hours) if t + pd.Timedelta(minutes=30) >= a and t + pd.Timedelta(minutes=30) < b]
        for n, k in enumerate(cols):
            c = gs.cell(rowi, 3 + k)
            c.fill = WARN_FILL if (rowi, k) in painted else fill  # 같은 줄에서 겹치면 빨강
            painted.add((rowi, k))
            if n == 0:
                c.value, c.font = label, Font(name=F, size=8, bold=True)
    rowi = 5
    for b in BERTHS:
        vs = [x for x in rows if x["r"]["선석"] == b]
        gs.cell(rowi, 1, b).font = BOLD
        gs.cell(rowi, 2, "터미널 예정").font = NORMAL
        gs.cell(rowi + 1, 2, "모델 예측").font = NORMAL
        for x in vs:
            r = x["r"]
            name = str(r["선박명"])[:10]
            paint(rowi, r["입항일시"], r["출항일시"], name, PLAN_FILL)
            paint(rowi + 1, x["berth_t"], x["dep_t"], name, MODEL_FILL)
        rowi += 3

    # ---------------- 시간대별 작업량 ----------------
    ls = wb.create_sheet("시간대별작업량", 3)
    ls["A1"], ls["A1"].font = "시간대별 예상 작업량 (모델 예측 작업시간 기준) → 크레인·인력 배치 참고", TITLE
    ls["A2"] = "선석별 값 = 그 시각에 작업 중인 배의 물량 ÷ 예측 작업시간 (box/시간). 필요 QC = 합계 ÷ QC 1기 처리량('가정' 시트)."
    ls["A2"].font = SMALL
    cols = ["시각"] + BERTHS + ["합계(box/h)", "필요 QC(대)", "필요 인원(명)", "작업 중 배(척)", "구분"]
    for j, h in enumerate(cols, start=1):
        c = ls.cell(4, j, h)
        c.font, c.fill, c.border = BOLD, HEAD_FILL, BOX
    for k, t in enumerate(hours):
        rr = 5 + k
        ls.cell(rr, 1, t.to_pydatetime()).number_format = HM
        mid = f"($A{rr}+1/48)"
        for j, b in enumerate(BERTHS, start=2):
            ls.cell(rr, j, f'=SUMPRODUCT(({rng("C")}="{b}")*({rng("S")}<={mid})*({rng("T")}>{mid})*{rng("H")}/{rng("R")})'
                    ).number_format = "0"
        ls.cell(rr, 6, f"=SUM(B{rr}:E{rr})").number_format = "0"
        ls.cell(rr, 7, f"=ROUNDUP(F{rr}/가정!$B$6,0)").number_format = "0"
        ls.cell(rr, 8, f"=G{rr}*가정!$B$7").number_format = "0"
        ls.cell(rr, 9, f'=SUMPRODUCT(({rng("S")}<={mid})*({rng("T")}>{mid}))').number_format = "0"
        ls.cell(rr, 10, f'=IF(G{rr}=0,"유휴",IF(G{rr}>=MAX($G$5:$G${4 + len(hours)})*0.8,"피크",""))')
        for j in range(1, 11):
            ls.cell(rr, j).font, ls.cell(rr, j).border = NORMAL, BOX
    ls.column_dimensions["A"].width = 13
    for j in range(2, 11):
        ls.column_dimensions[get_column_letter(j)].width = 11
    ls.freeze_panes = "B5"
    ch = BarChart()
    ch.type, ch.grouping, ch.overlap = "col", "stacked", 100
    ch.title, ch.y_axis.title = "선석별 시간당 작업량 (box/h)", "box/h"
    ch.add_data(Reference(ls, min_col=2, max_col=5, min_row=4, max_row=4 + len(hours)), titles_from_data=True)
    ch.set_categories(Reference(ls, min_col=1, min_row=5, max_row=4 + len(hours)))
    ch.height, ch.width = 9, 26
    ls.add_chart(ch, "L4")

    wb.calculation.fullCalcOnLoad = True
    wb.save(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--as-of", default=None, help='기준 시각. 안 주면 엑셀 파일명의 내보낸 시각 또는 지금')
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--out", default="오늘_접안예측표_v2.0.xlsx")
    a = ap.parse_args()
    path = a.path or latest_upload()
    if a.as_of:
        as_of = pd.Timestamp(a.as_of)
    else:
        import re
        m = re.search(r"(20\d{12})", os.path.basename(path))
        as_of = pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S").floor("min") if m else pd.Timestamp.now().floor("min")
    rows = build(path, as_of, a.days, a.out)
    print(f"{a.out}: {len(rows)}척 (작업중 {sum(x['kind'] == '작업중' for x in rows)})  기준 {as_of}")
    for x in rows:
        r = x["r"]
        print(f"  {r['선석']} {r['선박명'][:16]:16s} 예정 {r['입항일시']:%m/%d %H:%M} → 모델 {x['berth_t']:%m/%d %H:%M} "
              f"| 출항 예정 {r['출항일시']:%m/%d %H:%M} → 모델 {x['dep_t']:%m/%d %H:%M} {x.get('flag', '')}")


if __name__ == "__main__":
    main()
