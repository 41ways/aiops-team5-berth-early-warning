"""
애드온 업무 규칙·근거값 (v1.1). 숫자마다 출처를 단다. 확인 안 된 것은 CONFIRMED=False.
계산에 쓰는 값은 여기서만 가져다 쓴다 → 근거가 바뀌면 이 파일만 고치면 된다.
"""
RULES = {
    "cct_hours_before_berth": {
        "value": 10, "confirmed": True,
        "source": "HJNC 공지 'CCT(반입마감시간) 준수 및 지연 반입 청구 제도' — CCT(Cargo Closing Time, 접안 10시간 전), "
                  "수출 Dry 컨테이너, 2026.08.03 접안 모선부터. https://www.hjnc.co.kr/board/notice/detail?no=1381 "
                  "(엑셀 데이터에서도 미래 배 92%가 반입 마감 + 10시간)"},
    "cct_reset_if_delay_over_h": {
        "value": 12, "confirmed": False,
        "source": "PNC(부산신항 타 터미널) 규정: 접안이 12시간 이상 늦어질 때만 반입 마감을 다시 정함. HJNC는 미확인. "
                  "https://eascpcd.maersk.com/~/media_sc9/maersk/local-information/files/asia-pacific/korea/export/"
                  "busan-pnc-export-full-container-late-gate-in-container-cut-off-rule-implementation.pdf"},
    "min_gap_between_vessels_h": {
        "value": 2.0, "confirmed": True,
        "source": "국내(부산) 터미널 실증 논문: 예인선·도선·줄잡이에 접안·이안마다 약 1시간 → 배 사이 최소 2시간 "
                  "(Rescheduling Strategy for Berth Planning..., JMSE 2021, https://www.mdpi.com/2077-1312/9/5/527). "
                  "HJNC 실적도 같은 선석 번호에서 앞 배 출항 → 다음 배 접안 간격 하위 10% 1.8h, 25% 2.1h, 중앙값 2.7h"},
    "quay_cranes": {
        "value": 12, "confirmed": True,
        "source": "HJNC 장비 현황 'STS CRANE(G/C) 총 12기' https://www.hjnc.co.kr/terminal/equipment"},
    "quay_length_m_berths": {
        "value": (1100, 3), "confirmed": True,
        "source": "HJNC 시설 현황 '1.1km (3선석)' https://www.hjnc.co.kr/terminal/facility — 엑셀의 선석 번호(1B~4B)는 "
                  "물리 선석이 아니라 안벽 위치 구분. 같은 번호에 두 배가 겹치는 계획(미래 계획의 43%)이 흔함"},
    "qc_moves_per_hour": {
        "value": 25.1, "confirmed": True,
        "source": "부산항 안벽 크레인 시간당 25.1개 (KoreaScience CFKO200910348036286, 2009년 자료로 오래됨). "
                  "HJNC 실적 역산: 시간당 처리 99% 분위 326 box ÷ 12기 = 27"},
    "congestion_wait_h": {
        "value": 12, "confirmed": True,
        "source": "체선율 = 12시간 이상 묘박지 대기 선박 비율, 정부 목표 5% 이내 (김은수·김근섭, 한국항해항만학회지 40(1), 2016, "
                  "https://koreascience.or.kr/article/JAKO201609562999738.pdf)"},
    "berthing_wind_limit_ms": {
        "value": 13.5, "confirmed": False,
        "source": "대형 컨테이너선 접안: 평균 풍속 약 13.5 m/s 이하면 4,500HP 예인선 2척으로 가능, 보퍼트 6 이상 피할 것 "
                  "(해양환경안전학회지 13(2), 2007, https://koreascience.or.kr/article/JAKO200727500234381.do). 부산항 공식 기준 아님"},
    "shift_hours": {
        "value": None, "confirmed": False,
        "source": "부산항 터미널은 크레인당 3교대 인력 투입 사례(KoreaScience CFKO200910348032446) — 교대 시각은 미확인. "
                  "HJNC 교대 시각을 받으면 --shifts 07,15,23 처럼 넣는다"},
    # v2.5 추가 (2026-10-08 자료 수집, port-arrival-inputs/findings.md). 대시보드 선박 상세 표시용, 예측 계산에는 안 씀
    "tug_required_gt": {
        "value": 1000, "confirmed": True,
        "source": "부산항예선운영세칙(고시 제2025-104호) 제4조① 총톤수 1,000톤 이상 선박은 이·접안 때 예선 사용"},
    "tug_count_by_gt": {
        "value": ((5000, 1), (40000, 2), (70000, 2), (float("inf"), 3)), "confirmed": True,
        "source": "부산항예선운영세칙 별표1 (이·접안 보조장비 없는 선박): 5천 미만 1척, 5천~7만 2척, 7만 이상 3척"},
    "tug_request_lead_h": {
        "value": 24, "confirmed": True, "source": "부산항예선운영세칙 제5조① 사용개시 24시간 전 신청"},
    "pilot_request_lead_h": {
        "value": 3, "confirmed": True,
        "source": "도선약관 제3조·별표1 '부산 3시간 전' (2025-01-06 개정) http://www.busanpilot.co.kr/rule/terms"},
    "compulsory_pilot_gt": {
        "value": 500, "confirmed": True, "source": "부산항 도선구 도선안전절차 제4조 외항선 총톤수 500톤 이상 강제도선"},
    # v2.5 드리프트 판정·재학습 창 (addons/drift.py). 근거: 2026-10-08 실데이터 시뮬레이션 (HJNC·BNCT·PNIT)
    "drift_window_days": {
        "value": 14, "confirmed": True,
        "source": "터미널별 최근 14일 (정기선 주간 서비스 2회전). 척수 고정(21척)은 터미널마다 기간이 7.6~14일로 달라짐. "
                  "HJNC 7일 창 오경보 2%, 길수록 오경보↓·감지 지연↑ → 사용자 결정 14일"},
    "drift_min_ships": {
        "value": 15, "confirmed": True, "source": "14일 창에 정답 확정 배가 15척 미만이면 판정 보류 (PNIT 하루 1.5척 × 14일 ≈ 21척 기준 여유)"},
    "drift_bias_h": {
        "value": 1 / 3, "confirmed": True,
        "source": "창 안 평균 부호 오차 |실제 − 모델| ≥ 20분이면 드리프트. 기존 규칙(모델 MAE ≥ 터미널 MAE + 3분, RMSE > 60분)은 "
                  "모두가 같이 늦어지는 변화를 못 잡음: HJNC +30분 주입 감지 2% → 이 규칙 추가 시 96%, 오경보 2% 그대로"},
    "auto_promote_after_retrain": {
        "value": "margin", "confirmed": True,
        "source": "사용자 결정 2026-10-08: 재학습 모델이 확실히 좋으면 자동 승격, 아니면 후보(Staging)로만 두고 Production 은 이전 버전 유지. "
                  "확실히 좋음 = 게이트 통과 + 터미널 예정보다 좋음 + 운영 모델보다 promote_margin_h 이상 좋음. "
                  "후보는 관리 콘솔에서 수동 승격 가능. 환경변수 AUTO_PROMOTE=1 게이트만 넘으면 자동, 0 항상 수동"},
    "promote_margin_h": {
        "value": 1 / 60, "confirmed": True,
        "source": "자동 승격 최소 개선폭 1분. 재학습 방식 비교 실험(reports/retrain_strategy_v2.5.md)에서 방식 간 차이가 1분 안팎이라 그 이하는 우연과 구분 어려움"},
    "retrain_days": {
        "value": 28, "confirmed": True, "source": "재학습 데이터 최근 28일 (정기선 4회전, HJNC 약 77척·BNCT 약 63척·PNIT 약 42척)"},
    # v2.5 손해액·추천 행동(addons/impact.py)용 비용값. confirmed=False 는 출처 없는 가정 → 화면·API에 "가정"으로 표시하고
    # 요청 파라미터로 바꿔 넣을 수 있게 한다. 예측 계산에는 안 씀
    "vessel_class_by_gt": {
        "value": ((30000, "소형 ~2,500TEU"), (60000, "중형 ~4,250TEU"), (110000, "대형 ~8,500TEU"), (float("inf"), "초대형 13,000TEU+")), "confirmed": False,
        "source": "가정. 총톤수 → 선형 구간 (대략 총톤수 ÷ 11 ≈ TEU). 총톤수는 부산항만공사 입출항 집계(배 이름 매칭, HJNC 84%)"},
    "charter_usd_per_day_by_class": {
        "value": {"소형 ~2,500TEU": 43000, "중형 ~4,250TEU": 58000, "대형 ~8,500TEU": 79000, "초대형 13,000TEU+": 94750}, "confirmed": True,
        "source": "HARPEX 2026-10-02 정기용선료(6~12개월) 2,500TEU 43,000 / 4,250TEU 58,000 / 8,500TEU 79,000달러/일 (New ConTex 2026-10-01도 비슷). "
                  "13,000TEU+ 94,750은 공개 지수가 없어 8,500TEU에서 늘린 가정. port-arrival-inputs/loss_ship_side.md"},
    "anchor_fuel_t_per_day_by_class": {
        "value": {"소형 ~2,500TEU": 2.9, "중형 ~4,250TEU": 4.8, "대형 ~8,500TEU": 7.2, "초대형 13,000TEU+": 9.4}, "confirmed": False,
        "source": "가정. 묘박 중 보조전력 600~1,950kW × 24h × 200g/kWh (IMO 2020 접안 중 보조전력 평균 549kW, 범위 0~1,950에서 역산). 비용의 약 10%"},
    "lsmgo_usd_per_t": {
        "value": 1311, "confirmed": True,
        "source": "싱가포르 LSMGO 2026-10-05 (Ship & Bunker). 부산은 배출규제해역이라 묘박 중 황 0.1% 연료 → MGO 적용 (DNV). 부산 현지가 미확인"},
    "usd_krw": {"value": 1344.0, "confirmed": True, "source": "2026-10-05 프랑스 중앙은행 고시 유로/달러·유로/원 교차 계산"},
    "stevedoring_krw_per_20ft": {
        "value": 50867, "confirmed": True,
        "source": "항만하역요금표(2023-04-01 인가) 컨테이너 20피트 개당 선내 50,867원, 40피트는 180%. 신항 실제 계약요율은 더 낮을 수 있음 → 하한값으로 20피트만 사용"},
    "night_holiday_surcharge": {
        "value": 0.5, "confirmed": True,
        "source": "같은 요금표 p.6 야간·공휴일·일요일 +50% (위험물 갑류 +100%·을류 +60%는 지연과 무관). 근로기준법 56조 야간(22~06시) 가산 50%"},
    "truck_wait_krw_per_hour": {
        "value": 40000, "free_h": 1.0, "confirmed": True,
        "source": "2026 화물자동차 안전운임 고시(제2026-55호) 대기료 30분당 2만 원, 부두 내 1시간 초과분부터 (물류신문 인용 보도로 확인, 고시 별표 원문은 미확인)"},
    "transship_share_busan": {
        "value": 0.567, "confirmed": True, "source": "2025 부산항 2,488만 TEU 중 환적 1,410만 TEU (2026-01-29 보도, 계산값)"},
    "reefer_free_plugin_h": {
        "value": 48, "confirmed": True,
        "source": "수입 냉동 컨테이너 무료 전원 48시간, 이후 터미널이 직접 청구 (Hapag-Lloyd 한국 D&D 요율 2025). 터미널 전기료 단가 미확인"},
    "cranes_per_vessel": {
        "value": 4, "confirmed": False,
        "source": "가정. HJNC 안벽 크레인 12기 ÷ 3선석 = 선석당 4기 (장비 현황). 배별 실제 배정 대수는 비공개"},
    "gang_cost_krw_per_hour": {
        "value": 500000, "confirmed": False,
        "source": "가정 (출처 없음). 크레인 1기를 돌리는 작업조 1개(운전·신호·래싱·야드 연계)의 시간당 인건비. 터미널 원가 받으면 교체"},
    "overtime_multiplier": {
        "value": 1.5, "confirmed": True, "source": "근로기준법 제56조 연장근로 통상임금 50% 가산"},
    "reefer_share": {
        "value": 0.05, "confirmed": False,
        "source": "가정. 공개 스케줄에 냉동 컨테이너 수가 없어 물량의 5%로 둠. 실제 비율 받으면 교체"},
    "reefer_krw_per_box_hour": {
        "value": 3000, "confirmed": False, "source": "가정 (출처 없음). 냉동 컨테이너 1개 전원·감시 시간당 비용"},
}


def tugs_for(gt):
    """총톤수 → 예선 척수 (부산항예선운영세칙 별표1). 모르면 None, 의무 대상 아니면 0."""
    if gt is None:
        return None
    if gt < RULES["tug_required_gt"]["value"]:
        return 0
    return next(n for upper, n in RULES["tug_count_by_gt"]["value"] if gt < upper)


def v(key):
    return RULES[key]["value"]
