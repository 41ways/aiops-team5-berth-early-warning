"""
v2.5: 신항 5개 터미널 선석 스케줄 → HJNC 엑셀과 같은 칸 + `터미널` 태그 한 파일.

대시보드에서 이 파일 하나만 업로드하면 터미널별로 나눠 예측·검증한다 (addons/dashboard.py).
기존 파이프라인(/data/upload 검사, /predict/replay, 재학습)도 같은 칸이라 그대로 읽힌다.

터미널별로 데이터가 다르다 (2026-10-08 확인, port-arrival-inputs/busan_all_terminals.md)
  터미널 | 원본                       | 실제 접안          | 반입 마감(CCT)                      | 예측·검증
  HJNC   | 홈페이지 엑셀              | 완료 배는 실제     | 접안 10h 전, 정각 고정              | 가능
  PNIT   | 홈페이지 엑셀(HTML 표)     | 완료 배는 실제     | 예정 배 100% 접안 10h 전, 정각 고정 | 가능 (HJNC와 같은 규칙)
  BNCT   | 홈페이지 JSON (ETB/ATB)    | ATB                | ETB 10h 전 73%, 24h 전도 섞임       | 가능 (2026-07~09만). 예정 = ETB, 반입 마감 칸을 ETB − 10h로 맞춤
                                                                                                    (원본 CCT는 `반입 마감(원본)`). ETB가 나중에 고쳐졌는지는 미확인
  HPNT   | 홈페이지 엑셀(HTML 표)     | 한 칸              | 모든 배가 정확히 접안 12h 전        | 불가 → 실제 접안에 맞춰 다시 계산된 값이라 정답을 못 만듦
  DGT    | 홈페이지 JSON              | etb=atb 덮어씀     | 76%가 정확히 접안 12h 전            | 불가 (같은 이유)
  예측 불가 터미널은 `반입 마감일시`를 비워 학습·검증 샘플에서 빠지게 하고, 원본 값은 `반입 마감(원본)`에 남긴다.
  → 척수·물량(혼잡) 표시에는 쓴다.

작업 시작·완료가 없는 터미널(PNIT·HPNT·BNCT)은 접안·출항 시각으로 채우고 `작업시각=접안출항대체` 로 표시한다.
모선항차는 터미널끼리 겹칠 수 있어 앞에 터미널 이름을 붙인다 (HJNC는 원래 값 유지).

실행 (app/ 에서):
  python addons/terminals.py      → data/real/sinhang5_berth_schedule.csv
"""
import glob
import io
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = "data/real/terminals_raw"
HJNC = "data/real/hjnc_berth_schedule_20261008014718.xlsx"


def latest_hjnc() -> str:
    """data/real 의 HJNC 엑셀 중 가장 최근에 받은 것 (파일명 = 내보낸 시각, addons/refresh.py 가 받음)."""
    fs = sorted(glob.glob("data/real/hjnc_berth_schedule_*.xlsx"))
    return fs[-1] if fs else HJNC
OUT = "data/real/sinhang5_berth_schedule.csv"
COLS = ["No", "선석", "항로", "모선항차", "선박명", "선사항차", "접안", "선사", "반입 시작일시", "반입 마감일시", "입항일시",
        "출항일시", "작업 시작일시", "작업 완료일시", "양하", "선적", "S/H", "전배"]
LABEL_OK = {"HJNC": "Y", "PNIT": "Y", "BNCT": "Y", "HPNT": "N", "DGT": "N"}


def _ts(s):
    return pd.to_datetime(pd.Series(s).replace("", None), errors="coerce", format="mixed")


def hjnc(path=None) -> pd.DataFrame:
    d = pd.read_excel(path or latest_hjnc(), header=1)
    d.columns = [str(c).strip() for c in d.columns]
    d["작업시각"] = "원본"
    d["상태(원본)"] = d["작업 완료일시"].notna().map({True: "완료", False: "예정/작업중"})
    return d


def _pnit_like(name: str) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(RAW, f"*_{name.lower()}_*.xls")))
    t = pd.concat([max(pd.read_html(io.StringIO(open(f, encoding="utf-8", errors="ignore").read())), key=len) for f in fs])
    t = t.drop_duplicates(subset=["모선항차"], keep="last")
    berth, dep = _ts(t["접안(예정)일시"]), _ts(t["출항(예정)일시"])
    st = t["상태"].astype(str).str.upper()
    started = st.isin(["ARRIVED", "DEPARTED", "WORKING", "BERTHED"])
    d = pd.DataFrame({
        "선석": t["선석"], "항로": t["ROUTE"], "모선항차": name + "-" + t["모선항차"].astype(str), "선박명": t["선명"],
        "선사항차": t["선사항차"], "접안": t["선석"].astype(str).str.extract(r"\((\w)\)")[0], "선사": t["선사"],
        "반입 시작일시": _ts(t["반입시작일시"]) if "반입시작일시" in t else pd.NaT,
        "반입 마감일시": _ts(t["반입마감시한"]), "입항일시": berth, "출항일시": dep,
        "작업 시작일시": berth.where(started), "작업 완료일시": dep.where(st == "DEPARTED"),
        "양하": t["양하"], "선적": t["적하"], "S/H": t["Shift"], "전배": None,
        "작업시각": "접안출항대체", "상태(원본)": t["상태"]})
    return d.reset_index(drop=True)


def bnct() -> pd.DataFrame:
    f = sorted(glob.glob(os.path.join(RAW, "*_bnct_*.json")))[-1]
    t = pd.DataFrame(json.load(open(f)))
    atb, atd = _ts(t["ATBDATE"]), _ts(t["ATDDATE"])
    berth, dep = atb.fillna(_ts(t["ETBDATE"])), atd.fillna(_ts(t["ETDDATE"]))
    return pd.DataFrame({
        "선석": t["BERTHNO"], "항로": t["ROUTE"], "모선항차": "BNCT-" + t["VVD"] + "-" + t["VVD_YEAR"], "선박명": t["VSLNAME"],
        "선사항차": t["OPERVSLVVDIN"] + "/" + t["OPERVSLVVDOUT"], "접안": t["BERTHSIDE"], "선사": t["OPERATOR"],
        # BNCT 반입 마감은 ETB 10h 전(73%)과 24h 전이 섞여 있다 → 기록된 예정 접안(ETB)을 예정으로 쓰도록 ETB − 10h로 맞춘다
        "반입 시작일시": pd.NaT, "반입 마감일시": _ts(t["ETBDATE"]) - pd.Timedelta(hours=10), "입항일시": berth, "출항일시": dep,
        "작업 시작일시": atb, "작업 완료일시": atd, "양하": t["DISVAN"], "선적": t["LOADVAN"], "S/H": t["SHIFTVAN"], "전배": None,
        "작업시각": "접안출항대체", "상태(원본)": atd.notna().map({True: "DEPARTED", False: "ARRIVED/PLANNED"}),
        "반입 마감(원본)": _ts(t["CCTDATE"])})


def dgt() -> pd.DataFrame:
    f = sorted(glob.glob(os.path.join(RAW, "*_dgt_*.json")))[-1]
    t = pd.DataFrame(json.load(open(f))["vesselSchedules"])
    return pd.DataFrame({
        "선석": t["berthNo"], "항로": t["serviceLane"], "모선항차": "DGT-" + t["vesselCode"] + "-" + t["voyageYear"] + "-" + t["voyageSeq"],
        "선박명": t["vesselName"], "선사항차": t["inVoyage"] + "/" + t["outVoyage"], "접안": t["alongSide"], "선사": t["carrier"],
        "반입 시작일시": _ts(t["cyOpen"]), "반입 마감일시": _ts(t["cctDate"]), "입항일시": _ts(t["atb"]).fillna(_ts(t["etb"])),
        "출항일시": _ts(t["atd"]).fillna(_ts(t["etd"])), "작업 시작일시": _ts(t["workStartDate"]), "작업 완료일시": _ts(t["workEndDate"]),
        "양하": t["dischargeQty"], "선적": t["loadQty"], "S/H": t["shiftQty"], "전배": None,
        "작업시각": "원본", "상태(원본)": t["status"]})


def build() -> pd.DataFrame:
    parts = {"HJNC": hjnc(), "PNIT": _pnit_like("PNIT"), "HPNT": _pnit_like("HPNT"), "BNCT": bnct(), "DGT": dgt()}
    out = []
    for name, d in parts.items():
        d = d.copy()
        d["터미널"] = name
        d["예측검증"] = LABEL_OK[name]
        for c in ["반입 시작일시", "반입 마감일시", "입항일시", "출항일시", "작업 시작일시", "작업 완료일시"]:
            d[c] = pd.to_datetime(d[c], errors="coerce", format="mixed")
        if "반입 마감(원본)" not in d:
            d["반입 마감(원본)"] = d["반입 마감일시"]
        # 반입 마감이 실제 접안보다 늦은 기록은 마감이 나중에 다시 잡힌 것 → 예정으로 못 쓴다 (PNIT 2% 정도)
        late = d["반입 마감일시"] >= d["입항일시"]
        d.loc[late & d["작업 시작일시"].notna(), "반입 마감일시"] = pd.NaT
        if LABEL_OK[name] == "N":
            d["반입 마감일시"] = pd.NaT  # 정답(실제 − 예정)을 못 만드는 터미널은 학습·검증 샘플에서 뺀다
        for c in ["양하", "선적", "S/H"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
        out.append(d)
    df = pd.concat(out, ignore_index=True).sort_values("입항일시").reset_index(drop=True)
    df["No"] = range(1, len(df) + 1)
    return df[COLS + ["터미널", "예측검증", "반입 마감(원본)", "작업시각", "상태(원본)"]]


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M")
    print(OUT, len(df))
    print(df.groupby("터미널").agg(척수=("모선항차", "size"), 처음=("입항일시", "min"), 마지막=("입항일시", "max"),
                                  완료=("작업 완료일시", lambda s: int(s.notna().sum()))).to_string())
