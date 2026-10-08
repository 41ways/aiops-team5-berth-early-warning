"""
HJNC 선석 스케줄(엑셀 내보내기) → 학습용 데이터.

엑셀 컬럼 해석 (2026-10-08 내보내기 1,903척으로 확인):
  선석        : 1B~4B, OT
  항로        : 서비스 노선 코드 (68개)
  모선항차    : 배+항차 고유 ID (예: HOHA-2024-0003)
  선박명/선사 : 배 이름 / 선사 코드
  접안        : S(우현)/P(좌현) 접안 방향
  반입 시작일시: 수출 컨테이너 반입 시작. 접안 3~4일 전 날짜 00시
  반입 마감일시: 반입 마감. 미래 배는 '입항일시 − 10시간'(90%), 완료된 배는 정각 고정
  입항일시    : 미래 배 = 접안 예정, 완료된 배 = 실제 접안(분 단위 82%)
  출항일시    : 미래 배 = 출항 예정, 완료된 배 = 출항(실제)
  작업 시작/완료일시: 하역 작업 시작·완료 (완료된 배만)
  양하/선적/S/H: 내린 박스 / 실은 박스 / 선내 이동(Shifting) 박스. 미래 배는 계획 물량
  → 완료된 배의 '반입 마감 + 10시간' = 접안 약 10시간 전 시점의 터미널 접안 예정

학습 단위: 작업이 끝난 배 1척 (data/features.py)
  접안 오차(h) = 실제 접안 − 터미널 예정(반입 마감 + 10h)   ← 교수님 예제의 '종가' 자리
  처리 물량(box) = 양하 + 선적 + S/H                        ← 교수님 예제의 '거래량' 자리
  작업시간·생산성은 일정표(serving_app/schedule.py)에서 쓴다.
"""
import pandas as pd

TIME_COLS = ["반입 시작일시", "반입 마감일시", "입항일시", "출항일시", "작업 시작일시", "작업 완료일시"]
CUTOFF_LEAD_H = 10.0


def _h(a, b):
    return (b - a).dt.total_seconds() / 3600


def read_berth_excel(path: str) -> pd.DataFrame:
    """엑셀(첫 줄 제목, 둘째 줄 헤더) 또는 같은 컬럼의 CSV를 읽어 정리된 표로."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, header=1)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for c in TIME_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["양하", "선적", "S/H"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["box"] = df["양하"] + df["선적"] + df["S/H"]
    df["done"] = df["작업 완료일시"].notna()
    df["plan_berth_10h"] = df["반입 마감일시"] + pd.Timedelta(hours=CUTOFF_LEAD_H)
    df["work_h"] = _h(df["작업 시작일시"], df["작업 완료일시"])
    df["berth_h"] = _h(df["입항일시"], df["출항일시"])
    df["pre_h"] = _h(df["입항일시"], df["작업 시작일시"])     # 접안 → 작업 시작
    df["post_h"] = _h(df["작업 완료일시"], df["출항일시"])    # 작업 완료 → 출항
    df["prod"] = df["box"] / df["work_h"]
    return df


def completed_calls(df: pd.DataFrame, min_box: int = 50, min_work_h: float = 1.0) -> pd.DataFrame:
    """학습에 쓸 완료 선박. 물량·작업시간이 너무 작은 기록(빈 배 이동 등)은 뺀다."""
    d = df[df["done"] & (df["box"] >= min_box) & (df["work_h"] >= min_work_h)].copy()
    d["prod"] = d["box"] / d["work_h"]
    d = d[(d["prod"] > 5) & (d["prod"] < 300)]  # 기록 오류로 보이는 극단값 제외 (이유: 문서에 기록)
    return d.sort_values("작업 완료일시").reset_index(drop=True)
