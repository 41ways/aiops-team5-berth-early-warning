"""
v2.5: 접안 지연 → 예상 손실액 기준표 (러프, 근거 확인된 값만 합계에 넣음).

  GET /v25/loss-table?delay_h=2   선형 4구간 × 손실 항목 (합계는 확인된 값만), 적재 종류별 영향, 권고 조치

손실 항목 (누가 부담하나 · 근거 · 합계 포함 여부)
  A 배 대기       (선사)  지연 h × (하루 용선료 ÷ 24 + 묘박 연료) × 환율    HARPEX 2026-10-02, LSMGO 2026-10-05     포함
  B 야간·휴일 할증 (선사·화주) 밀린 작업이 22~06시·휴일로 넘어가면 밀린 박스 × 20피트 하역료 × 50%    항만하역요금표 2023      조건부(따로 표시)
  C 다음 배 대기   (다음 배 선사) 같은 선석 다음 배가 밀리면 그 배에 A 를 다시 적용                         같은 근거                조건부
  D 트럭 대기      (운송사·화주) 1시간 넘게 기다린 반출 차량 1대당 (대기 − 1h) × 4만 원                   2026 안전운임 대기료     단가만
  E 작업조 대기    (터미널) 시간당 인건비 공개 자료 없음                                                   미확인                   제외
  냉동은 수입 무료 전원 48시간 안이면 2시간 지연 추가 비용 없음, 위험물 할증(갑 +100%·을 +60%)은 지연과 무관

선형별 물량·처리 속도는 HJNC 완료 선박(2026-10-08 11시 파일) 중앙값. 총톤수 → 선형은 가정 구간.
"""
from fastapi.routing import APIRoute

from addons.rules import RULES
from serving_app.main import app

# HJNC 완료 선박 중앙값 (총톤수 매칭 84%, data/real/hjnc_berth_schedule_20261008110003.xlsx 로 계산)
CLASS_STATS = {
    "소형 ~2,500TEU": {"n": 563, "box": 689, "unload": 308, "load": 328, "work_h": 12.1, "box_per_h": 53.4},
    "중형 ~4,250TEU": {"n": 223, "box": 950, "unload": 467, "load": 583, "work_h": 13.7, "box_per_h": 72.0},
    "대형 ~8,500TEU": {"n": 337, "box": 1419, "unload": 544, "load": 947, "work_h": 17.3, "box_per_h": 83.9},
    "초대형 13,000TEU+": {"n": 362, "box": 3594, "unload": 1960, "load": 1232, "work_h": 32.9, "box_per_h": 113.0},
}
DEFAULT_CLASS = "중형 ~4,250TEU"


def v(k):
    return RULES[k]["value"]


def vessel_class(gt) -> str:
    if gt is None:
        return DEFAULT_CLASS
    return next(c for upper, c in v("vessel_class_by_gt") if gt < upper)


def carrier_krw_per_hour(cls: str) -> dict:
    charter = v("charter_usd_per_day_by_class")[cls] / 24
    fuel = v("anchor_fuel_t_per_day_by_class")[cls] / 24 * v("lsmgo_usd_per_t")
    fx = v("usd_krw")
    return {"charter_usd_h": round(charter), "fuel_usd_h": round(fuel), "total_krw_h": round((charter + fuel) * fx),
            "charter_confirmed": cls != "초대형 13,000TEU+"}


def night_surcharge_krw(boxes: float) -> int:
    return round(boxes * v("stevedoring_krw_per_20ft") * v("night_holiday_surcharge"))


def truck_wait_krw(trucks: int, wait_h: float) -> int:
    r = RULES["truck_wait_krw_per_hour"]
    return round(trucks * max(0.0, wait_h - r["free_h"]) * r["value"])


CARGO = [
    {"kind": "일반(드라이)", "impact": "하역 순서·시간이 통째로 밀림. 밀린 작업이 22~06시나 휴일로 넘어가면 하역료 50% 할증",
     "action": "작업조 투입을 예측 접안 30분 전으로 늦추고, 비는 시간은 접안 중인 다른 배·야드 작업에 배치"},
    {"kind": "수입 반출(트럭)", "impact": "양하가 늦어져 반출 예약 차량이 대기. 부두 안 1시간 넘으면 30분당 2만 원 대기료",
     "action": "반출 예약 시간대를 예측 양하 시각 뒤로 옮기도록 운송사에 알림 (VBS 예약 변경)"},
    {"kind": "환적 (부산항 물량의 약 57%)", "impact": "연결 선박 출항 전까지 옮겨 실어야 함. 놓치면 다음 항차까지 장치 (비용 공개 자료 없음)",
     "action": "연결 선박 출항 시각과 예측 양하 시각 사이 여유 확인, 여유가 작은 박스부터 먼저 양하"},
    {"kind": "냉동", "impact": "수입 무료 전원 48시간 안이라 2시간 지연은 추가 비용 거의 없음. 화물 품질 위험은 전원 연결 대기 시간",
     "action": "전원 소켓과 점검 인력을 예측 접안 시각에 맞춰 준비"},
    {"kind": "위험물", "impact": "하역료 할증(갑류 +100%, 을류 +60%)은 지연과 무관. 직반출입 차량은 부두에 오래 머물 수 없음 (부산 체류 제한 시간 미확인)",
     "action": "직반출입 위험물 차량 도착 시각을 예측 접안 기준으로 다시 잡기"},
]


def actions_for(delay_h: float) -> list:
    acts = []
    if delay_h >= 0.5:
        acts.append(("터미널 운영사", f"작업조 투입을 예측 접안 30분 전으로 늦추기 ({delay_h:g}시간 대기 방지)"))
    if delay_h >= 1:
        acts.append(("선사·대리점", "도선(3시간 전)·예선(24시간 전) 신청 시각을 예측 접안 기준으로 다시 맞추기"))
        acts.append(("운송사", "수입 반출 예약 시간대를 예측 양하 시각 뒤로 옮기기"))
    if delay_h >= 2:
        acts.append(("터미널 운영사", "같은 선석 다음 배 접안이 2시간 간격을 못 지키면 다른 선석 검토 또는 다음 배 선사에 감속 입항 안내"))
        acts.append(("터미널 운영사", "작업이 22~06시로 넘어가면 야간 작업조 편성 확인 (하역료·인건비 50% 가산 구간)"))
    if delay_h >= 4:
        acts.append(("터미널 운영사·항만공사", "선석 재배정 검토, 환적 연결이 끊길 박스 목록 선사와 공유"))
    return [{"who": w, "what": t} for w, t in acts]


def loss_table(delay_h: float = 2.0):
    rows = []
    for cls, st in CLASS_STATS.items():
        c = carrier_krw_per_hour(cls)
        shifted = round(st["box_per_h"] * delay_h)
        a = round(c["total_krw_h"] * delay_h)
        b = night_surcharge_krw(shifted)
        rows.append({"class": cls, "typical_box": st["box"], "typical_unload": st["unload"], "typical_load": st["load"],
                     "box_per_h": st["box_per_h"], "work_h": st["work_h"],
                     "carrier_krw_per_h": c["total_krw_h"], "charter_usd_h": c["charter_usd_h"], "fuel_usd_h": c["fuel_usd_h"],
                     "A_vessel_wait_krw": a, "B_night_surcharge_if_night_krw": b, "boxes_shifted": shifted,
                     "C_next_vessel_wait_krw_if_blocked": a,
                     "D_truck_example_krw": truck_wait_krw(10, delay_h),
                     "total_confirmed_krw": a, "total_if_night_krw": a + b,
                     "charter_confirmed": c["charter_confirmed"],
                     "transship_box_est": round(st["box"] * v("transship_share_busan"))})
    return {"delay_h": delay_h, "rows": rows, "cargo": CARGO, "actions": actions_for(delay_h),
            "excluded": [{"item": "E 작업조 대기 (터미널 인건비)", "why": "하역 근로자 시간당 임금·크레인당 작업조 공식 기준 공개 자료 없음 → 터미널 원가 받으면 추가"},
                         {"item": "환적 연결 실패 비용", "why": "옮겨 싣기·장치 비용 공개 자료 없음"},
                         {"item": "냉동 전기료 (48시간 넘을 때)", "why": "신항 터미널 플러그인 단가 미확인"}],
            "notes": ["A 는 선사가 실제로 쓰는 돈(용선료·연료). 터미널 수입 손실이 아님",
                      "B 는 밀린 작업이 야간·휴일로 넘어갈 때만. 20피트 하역료(하한)로 계산, 40피트는 180%",
                      "D 는 예시: 반출 차량 10대가 지연 시간만큼 기다릴 때 (1시간까지 무료)",
                      "초대형 용선료는 공개 지수가 없어 가정값 포함",
                      "선형별 물량·처리 속도는 HJNC 완료 선박 중앙값"],
            "sources": {k: RULES[k]["source"] for k in ["charter_usd_per_day_by_class", "anchor_fuel_t_per_day_by_class", "lsmgo_usd_per_t",
                                                     "usd_krw", "stevedoring_krw_per_20ft", "night_holiday_surcharge",
                                                     "truck_wait_krw_per_hour", "transship_share_busan", "reefer_free_plugin_h",
                                                     "vessel_class_by_gt"]}}


app.router.routes.insert(0, APIRoute("/v25/loss-table", loss_table, methods=["GET"]))
