"""
애드온 실행: 선석 체류시간 분해 + 수용 능력 계산 → 엑셀 한 개 (기존 모델 파일은 건드리지 않음)

  python addons/make_ops_report.py [HJNC 엑셀] [--as-of "2026-10-08 01:47"] [--days 3] [--qc 12] [--qc-rate 25]
  → 운영계획표_addon_v1.0.xlsx, reports/ops_addon_v1.0/*.csv

--qc      : 터미널이 쓸 수 있는 안벽 크레인(QC) 대수 — 가정값, 터미널 실제 값으로 바꿀 것
--qc-rate : QC 1대의 시간당 처리 박스 — 가정값 (예측표 '가정' 시트와 같은 25)
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from addons.berth_time import breakdown, daily_capacity, summaries
from addons.capacity import MIN_GAP, N_BERTHS, historical_capacity, hourly_load, vessel_plan, warnings
from addons.rules import RULES, v as rule
from data.hjnc import read_berth_excel
from data.storage import latest_upload

F = "Arial"
BOLD, NORMAL, BLUE, SMALL = Font(name=F, bold=True), Font(name=F), Font(name=F, color="0000FF"), Font(name=F, size=9, color="555555")
TITLE = Font(name=F, bold=True, size=14)
HEAD = PatternFill("solid", fgColor="DDE5F0")
WARN = PatternFill("solid", fgColor="F4B6B6")
AMBER = PatternFill("solid", fgColor="FBE3B0")
BOX = Border(*(Side(style="thin", color="BBBBBB"),) * 4)
DT = "mm/dd hh:mm"


def table(ws, r0, df, widths=None, fmt=None):
    for j, c in enumerate(df.columns, 1):
        cell = ws.cell(r0, j, str(c))
        cell.font, cell.fill, cell.border = BOLD, HEAD, BOX
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for i, row in enumerate(df.itertuples(index=False), r0 + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()
            c = ws.cell(i, j, v)
            c.font, c.border = NORMAL, BOX
            if hasattr(v, "hour"):
                c.number_format = DT
            elif fmt and df.columns[j - 1] in fmt:
                c.number_format = fmt[df.columns[j - 1]]
    for j, w in enumerate(widths or [], 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    return r0 + len(df) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--qc", type=int, default=rule("quay_cranes"), help="HJNC STS 12기 (addons/rules.py)")
    ap.add_argument("--qc-rate", type=float, default=rule("qc_moves_per_hour"), help="부산항 크레인 시간당 25.1개 (rules.py)")
    ap.add_argument("--shifts", default=None, help='교대 시작 시각, 예: "07,15,23". 확인된 값만 넣을 것')
    ap.add_argument("--out", default="운영계획표_addon_v1.0.xlsx")
    a = ap.parse_args()
    path = a.path or latest_upload()
    m = re.search(r"(20\d{12})", os.path.basename(path))
    as_of = pd.Timestamp(a.as_of) if a.as_of else (pd.to_datetime(m.group(1), format="%Y%m%d%H%M%S").floor("min") if m else pd.Timestamp.now().floor("min"))

    df = read_berth_excel(path)
    d = breakdown(df)
    S = summaries(d)
    cap = daily_capacity(df)
    recent_wait = d.sort_values("입항일시").groupby("선석").tail(30).groupby("선석")["wait_start_h"].mean().to_dict()

    v, end = vessel_plan(path, as_of, a.days)
    load = hourly_load(v, as_of, end, a.qc_rate)
    load_late = hourly_load(v, as_of, end, a.qc_rate, late=True)
    hist = historical_capacity(df)
    W = warnings(v, load, load_late, hist, recent_wait)

    os.makedirs("reports/ops_addon_v1.0", exist_ok=True)
    for name, x in (("berth_time_by_vessel", d), ("berth_time_by_month", S["by_month"]), ("daily_capacity", cap),
                    ("vessel_plan", v), ("hourly_load", load), ("hourly_load_late", load_late), ("warnings", W)):
        x.to_csv(f"reports/ops_addon_v1.0/{name}.csv", encoding="utf-8-sig")

    wb = Workbook()
    # ---- 가정 ----
    ga = wb.active
    ga.title = "가정"
    ga["A1"], ga["A1"].font = "가정값 (파란 글씨는 바꿔도 됨)", TITLE
    rows = [("선석 번호 수", N_BERTHS, "엑셀 1B~4B (물리 선석은 1.1km 3선석, HJNC 시설 현황)"),
            ("보유 안벽 크레인(QC) 대수", a.qc, "HJNC 장비 현황 STS 12기"),
            ("QC 1대 시간당 처리 (box/h)", a.qc_rate, "부산항 25.1개/h (2009 논문, 오래됨) — HJNC 실적 역산 27"),
            ("작업이 늦어질 때 배수", round(float(v["늦을때_작업완료"].sub(v["예측작업시작"]).dt.total_seconds().div(3600).div(v["작업시간_h"]).median()), 2),
             "검증 구간 작업시간 80% 범위 위쪽 (residual_band.json)"),
            ("기준 시각", as_of.to_pydatetime(), os.path.basename(path)),
            ("과거 실적 처리 한계 (box/h, 99% 분위)", round(hist["box_h_p99"]), "2025-01 이후 실제 작업 기록. '처리 한계 초과' 경고의 기준"),
            ("과거 실적 보통 처리량 (box/h, 중앙값)", round(hist["box_h_p50"]), "같은 기록의 중앙값")]
    for i, (k, val, s) in enumerate(rows, 3):
        ga.cell(i, 1, k).font = NORMAL
        c = ga.cell(i, 2, val)
        c.font = BLUE if i in (4, 5) else NORMAL
        if i == 7:
            c.number_format = "yyyy-mm-dd hh:mm"
        ga.cell(i, 3, s).font = SMALL
    ga["A11"] = "경고 목록은 스크립트 실행 시점의 가정값으로 계산한 결과다. QC 대수를 바꾸면 '시간대별부하'의 여유 열은 바로 바뀌지만 경고 목록은 다시 실행해야 바뀐다."
    ga["A11"].font = SMALL
    for col, w in zip("ABC", (30, 18, 60)):
        ga.column_dimensions[col].width = w

    # ---- 업무규칙 근거 ----
    rr = wb.create_sheet("업무규칙_근거")
    rr["A1"], rr["A1"].font = "계산에 쓴 업무 규칙과 출처 (addons/rules.py)", TITLE
    rdf = pd.DataFrame([{"항목": k, "값": str(x["value"]), "확인": "확인" if x["confirmed"] else "미확인", "출처": x["source"]}
                        for k, x in RULES.items()])
    table(rr, 3, rdf, [26, 12, 8, 140])
    rr.cell(len(rdf) + 6, 1, "여유 기준 (이 애드온의 권장)").font = BOLD
    rec = [("배 사이 최소 간격", f"{MIN_GAP:.0f}시간 (도선·예선·줄잡이, 논문 + HJNC 실적 하위 10% 1.8h)"),
           ("출항 지연 대비 (80%)", "예측 작업시간 × 0.32 만큼 더 (작업시간 80% 범위 위쪽). 테스트에서 출항이 예측보다 늦은 정도 80% 분위 4.1h"),
           ("출항 지연 대비 (90%)", "테스트 90% 분위 6.4h, 95% 분위 9.1h — 대형선·혼잡기에는 90% 기준 권장"),
           ("접안 지연 대비", "모델 기준 90% 분위 0.4h (터미널 예정 기준 0.7h)"),
           ("다음 배 접안 가능 시각 (80% 안전)", "예측 작업시작 + 작업시간 × 1.32 + 작업완료→출항 + 최소 간격 2h → '배별계획'의 80%안전_다음배접안가능")]
    for i, (k, val) in enumerate(rec, len(rdf) + 7):
        rr.cell(i, 1, k).font = NORMAL
        rr.cell(i, 2, val).font = NORMAL

    # ---- 요약 ----
    sm = wb.create_sheet("요약", 0)
    sm["A1"], sm["A1"].font = f"운영 계획 요약 (기준 {as_of:%Y-%m-%d %H:%M}, ~{(end - pd.Timedelta(minutes=1)):%m/%d})", TITLE
    sm["A3"], sm["A3"].font = "앞으로 며칠 경고", BOLD
    r = 4
    if W.empty:
        sm.cell(r, 1, "경고 없음").font = NORMAL
        r += 1
    else:
        for k, n in W.groupby(["종류", "시나리오"]).size().items():
            sm.cell(r, 1, f"{k[0]} ({k[1]})").font = NORMAL
            sm.cell(r, 2, int(n)).font = NORMAL
            r += 1
    pk = load.loc[load["시간당처리_box"].idxmax()]
    sm.cell(r + 1, 1, "작업량이 가장 많은 시각 (예측대로)").font = NORMAL
    sm.cell(r + 1, 2, f"{pk['시각']:%m/%d %H}시 · {int(pk['시간당처리_box'])} box/h · QC {int(pk['필요QC'])}대 · 선석 {int(pk['차있는선석'])}척")
    sm.cell(r + 2, 1, "선석이 다 찬 시간 (예측대로 / 늦어질 때)").font = NORMAL
    sm.cell(r + 2, 2, f"{int((load['차있는선석'] >= N_BERTHS).sum())}h / {int((load_late['차있는선석'] >= N_BERTHS).sum())}h")
    r += 4
    sm.cell(r, 1, "과거 실적: 선석 체류시간은 어디에 쓰였나").font = BOLD
    for i, (k, val) in enumerate(S["headline"].items(), r + 1):
        sm.cell(i, 1, k).font = NORMAL
        sm.cell(i, 2, val).font = NORMAL
    r = r + len(S["headline"]) + 2
    sm.cell(r, 1, "과거 실적: 하루 수용 능력").font = BOLD
    c26 = cap[cap.index >= "2026-01-01"]
    facts = [("하루 평균 도착 (2026)", round(float(c26["도착척수"].mean()), 2)), ("하루 최대 도착 (2026)", int(c26["도착척수"].max())),
             ("5척 이상 도착한 날 (2026)", int((c26["도착척수"] >= 5).sum())), ("평균 선석 점유율 (2026)", f"{c26['점유율'].mean() * 100:.0f}%"),
             ("4개 선석이 다 찬 시간 비중 (2026)", f"{c26['만석시간_h'].sum() / (len(c26) * 24) * 100:.1f}%"),
             ("점유율 90% 넘은 날 (2026)", int((c26["점유율"] >= 0.9).sum()))]
    for i, (k, val) in enumerate(facts, r + 1):
        sm.cell(i, 1, k).font = NORMAL
        sm.cell(i, 2, val).font = NORMAL
    sm.cell(r + len(facts) + 2, 1, "해석: HJNC는 하루 총량으로는 여유가 있다(점유율 60%대). 문제는 특정 날·선석에 몰리는 것과, 배가 붙은 뒤 "
                                   "작업 시작 전·출항 전에 선석이 노는 시간이다.").font = SMALL
    sm.column_dimensions["A"].width = 42
    sm.column_dimensions["B"].width = 60

    # ---- 경고 ----
    ws = wb.create_sheet("경고", 1)
    ws["A1"], ws["A1"].font = "경고 목록 (시작 시각 순)", TITLE
    ws["A2"] = "빨강 = 선석 부족·처리 한계 초과·같은 선석 겹침, 노랑 = 도착 몰림·접안 후 작업 대기 위험. '작업이 늦어질 때' = 작업시간 × 80% 범위 위쪽 배수"
    ws["A2"].font = SMALL
    end_row = table(ws, 3, W, [18, 16, 13, 13, 100])
    for i in range(4, end_row):
        kind = ws.cell(i, 1).value
        ws.cell(i, 1).fill = WARN if kind in ("선석 부족", "처리 한계 초과", "같은 선석 겹침") else AMBER

    # ---- 배별계획 ----
    vp = wb.create_sheet("배별계획", 2)
    vp["A1"], vp["A1"].font = "배별 계획 (기존 모델 예측: 접안 보정 + 작업시간 추정)", TITLE
    table(vp, 3, v, [7, 5, 18, 16, 6, 7, 12, 12, 12, 12, 12, 12, 12, 12, 9, 9])

    # ---- 시간대별부하 (여유는 수식: 가정 시트를 바꾸면 바로 반영) ----
    hl = wb.create_sheet("시간대별부하", 3)
    hl["A1"], hl["A1"].font = "시간대별 부하: 예측대로 vs 작업이 늦어질 때", TITLE
    cols = ["시각", "차있는선석", "작업중인배", "시간당처리_box", "필요QC(환산)", "선석여유", "처리여유_box_h",
            "늦을때_차있는선석", "늦을때_시간당처리_box", "늦을때_필요QC", "늦을때_처리여유"]
    for j, c in enumerate(cols, 1):
        cell = hl.cell(3, j, c)
        cell.font, cell.fill, cell.border = BOLD, HEAD, BOX
    for i, (x, y) in enumerate(zip(load.itertuples(index=False), load_late.itertuples(index=False)), 4):
        vals = [x.시각.to_pydatetime(), x.차있는선석, x.작업중인배, x.시간당처리_box, f"=ROUNDUP(D{i}/가정!$B$5,0)",
                f"=가정!$B$3-B{i}", f"=가정!$B$8-D{i}", y.차있는선석, y.시간당처리_box, f"=ROUNDUP(I{i}/가정!$B$5,0)", f"=가정!$B$8-I{i}"]
        for j, val in enumerate(vals, 1):
            c = hl.cell(i, j, val)
            c.font, c.border = NORMAL, BOX
            if j == 1:
                c.number_format = DT
    last = 3 + len(load)
    from openpyxl.formatting.rule import CellIsRule
    for col in ("F", "G", "K"):
        hl.conditional_formatting.add(f"{col}4:{col}{last}", CellIsRule(operator="lessThan", formula=["0"], fill=WARN))
    for j, w in enumerate([13, 10, 10, 13, 9, 9, 9, 14, 16, 12, 12], 1):
        hl.column_dimensions[get_column_letter(j)].width = w
    hl.freeze_panes = "B4"

    # ---- 체류시간 분해 ----
    bt = wb.create_sheet("체류시간분해", 4)
    bt["A1"], bt["A1"].font = "과거 선석 체류시간 분해 (작업 끝난 배, 물량 50 box 이상)", TITLE
    bt["A2"] = "작업시작대기 = 접안 → 작업 시작, 출항대기 = 작업 끝 → 출항(확정 출항 시각 기준, 대략값), 생산성손실 = 그 달 중앙 생산성 대비 더 걸린 시간"
    bt["A2"].font = SMALL
    r = 4
    for title, x in (("월별", S["by_month"]), ("선석별", S["by_berth"]), ("항로별 (10척 이상, 작업시작대기 긴 순)", S["by_route"])):
        bt.cell(r, 1, title).font = BOLD
        r = table(bt, r + 1, x.reset_index()) + 2
    bt.cell(r, 1, "선석이 가장 오래 논 배 15척 (작업시작대기 + 출항대기)").font = BOLD
    table(bt, r + 1, S["worst"])
    for j, w in enumerate([12, 16, 16, 8, 14, 10, 12, 12, 12, 10], 1):
        bt.column_dimensions[get_column_letter(j)].width = max(w, 11)

    # ---- 과거 하루별 ----
    dc = wb.create_sheet("과거_하루별", 5)
    dc["A1"], dc["A1"].font = "과거 하루별 도착·선석 점유", TITLE
    table(dc, 3, cap.reset_index().rename(columns={"index": "날짜"}), [12, 10, 12, 10, 12], fmt={"점유율": "0.0%"})
    for row in dc.iter_rows(min_row=4, max_col=1):
        row[0].number_format = "yyyy-mm-dd"

    if a.shifts:   # 확인된 교대 시각이 있을 때만 교대별로 묶는다
        starts = sorted(int(h) for h in a.shifts.split(","))
        def shift_of(t):
            hs = [h for h in starts if t.hour >= h]
            st = max(hs) if hs else starts[-1]
            day = t.normalize() if hs else t.normalize() - pd.Timedelta(days=1)
            return f"{day:%m/%d} {st:02d}시~"
        sh = wb.create_sheet("교대별", 4)
        sh["A1"], sh["A1"].font = f"교대별 부하 (교대 시작 {a.shifts}시)", TITLE
        g = pd.DataFrame({"교대": load["시각"].map(shift_of), "선석": load["차있는선석"], "box_h": load["시간당처리_box"],
                          "늦을때_box_h": load_late["시간당처리_box"]})
        out = g.groupby("교대", sort=False).agg(최대_차있는선석=("선석", "max"), 최대_시간당처리_box=("box_h", "max"),
                                                교대_총처리_box=("box_h", "sum"), 늦을때_최대_box_h=("늦을때_box_h", "max")).reset_index()
        out["필요QC_최대"] = (out["최대_시간당처리_box"] / a.qc_rate).apply(lambda x: int(-(-x // 1)))
        table(sh, 3, out, [16, 14, 16, 14, 16, 12])

    wb.calculation.fullCalcOnLoad = True
    wb.save(a.out)
    print(f"{a.out} 저장. 배 {len(v)}척, 경고 {len(W)}건")
    print(W.groupby(["종류", "시나리오"]).size().to_string() if not W.empty else "경고 없음")
    print("headline:", S["headline"])


if __name__ == "__main__":
    main()
