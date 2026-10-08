"""
v2.5: 예측 지연 → 예상 손해액 → 추천 행동 (규칙 기반 v0, 학습 모델 아님).

  GET /v25/impact?as_of=&horizon_h=48&terminal=ALL[&cranes=]
    vessels  : 배마다 접안 지연·작업(출항) 지연·다음 배 대기 → 손해 항목별 금액(원)과 누가 부담하나, 추천 행동과 아낄 수 있는 금액
    staffing : 터미널·시간대별 필요 작업조 수 (터미널 예정 기준 vs 예측 기준) → 줄일 시간대 / 더 필요한 시간대
    actions  : 전체 추천 행동을 아낄 수 있는 금액 순으로
    assumptions : 쓴 비용값과 출처. confirmed=false 는 출처 없는 가정이라 파라미터로 바꿔 넣는다

입력은 기존 계산을 그대로 쓴다
  접안 예측 : /predict/next-day (모델 A). addons/ais_eta.py 가 있으면 AIS 앙상블 접안 예측을 우선
  출항 예측 : WorkTimeEstimator (api.next_day 안)
  다음 배 대기 : /schedule/reschedule (같은 선석 앞 배 예측 출항 + 2시간 간격)

손해 항목 (부담 주체) — 단가·근거는 addons/loss.py, docs/손실액_기준표_v2.5.md
  vessel_wait      (선사)       (작업 지연 + 앞 배 때문에 기다리는 시간) × 선형별 시간당 비용(용선료 + 묘박 연료)
  night_surcharge  (선사·화주)  예측 작업이 터미널 예정보다 22~06시에 더 걸치는 시간 × 선형별 시간당 박스 × 20피트 하역료 × 50%
  labor_idle_gang_h·overtime_gang_h (터미널) 작업조·시간. 하역 인건비 단가 공개 자료가 없어 금액으로 바꾸지 않음
"""
import pandas as pd
from fastapi.routing import APIRoute

from addons import api
from addons.rules import RULES

GRADE_H = 0.5


def _assumptions(p: dict) -> list:
    keys = ["charter_usd_per_day_by_class", "anchor_fuel_t_per_day_by_class", "lsmgo_usd_per_t", "usd_krw", "vessel_class_by_gt",
            "stevedoring_krw_per_20ft", "night_holiday_surcharge", "cranes_per_vessel", "min_gap_between_vessels_h"]
    out = []
    for k in keys:
        r = RULES[k]
        v = p.get(k, r["value"])
        if isinstance(v, tuple):  # 구간표: 마지막 상한 inf 는 JSON에 못 실어 null 로
            v = [[None if u == float("inf") else u, w] for u, w in v]
        out.append({"key": k, "value": v,
                    "confirmed": bool(r["confirmed"]) and k not in p, "overridden": k in p, "source": r["source"]})
    return out


def _night_h(a: pd.Timestamp, b: pd.Timestamp) -> float:
    """[a, b) 안에서 22~06시에 걸친 시간."""
    if b <= a:
        return 0.0
    hrs = pd.date_range(a.floor("h"), b, freq="15min", inclusive="left")
    return sum(1 for t in hrs if (t.hour >= 22 or t.hour < 6) and a <= t < b) * 0.25


def _ais_berths(as_of, horizon_h) -> dict:
    """AIS 앙상블 모듈이 있으면 call_id → 앙상블 접안 예측(문자열). 없거나 실패하면 빈 dict."""
    try:
        from addons import ais_eta as ais  # noqa: F401
        fn = getattr(ais, "ais_eta", None) or getattr(ais, "ais_eta_endpoint", None)
        if fn is None:
            return {}
        res = fn(as_of=as_of, horizon_h=horizon_h)
        out = {}
        for v in res.get("vessels", []):
            cid = v.get("call_id") or v.get("call")
            eb = v.get("ensemble") or v.get("ensemble_berth")
            if cid and eb and (v.get("w") or v.get("weight") or 0) > 0:
                out[cid] = eb
        return out
    except Exception:
        return {}


def impact(as_of: str | None = None, horizon_h: float = 48.0, terminal: str = "ALL", cranes: int | None = None):
    from addons import loss as L
    p = {k: v for k, v in {"cranes_per_vessel": cranes}.items() if v is not None}
    C = int(p.get("cranes_per_vessel", RULES["cranes_per_vessel"]["value"]))
    G = None  # 작업조 시간당 인건비: 공개 자료 없음 → 금액 대신 작업조·시간으로만

    nd = api.next_day(as_of, horizon_h)
    rs = {v["call_id"]: v for v in api.reschedule(as_of, horizon_h)["vessels"]}
    ais = _ais_berths(as_of, horizon_h)
    t0 = pd.Timestamp(nd["as_of"])

    vessels, actions = [], []
    for v in nd["vessels"]:
        if terminal != "ALL" and v["terminal"] != terminal:
            continue
        plan_b = pd.Timestamp(v["terminal_planned_berth"])
        src = "ais_ensemble" if v["call_id"] in ais else "model_a"
        pred_b = pd.Timestamp(ais.get(v["call_id"], v["predicted_berth"]))
        pred_d = pd.Timestamp(v["predicted_departure"])
        plan_d = pd.Timestamp(v["terminal_planned_departure"]) if v["terminal_planned_departure"] else None
        r = rs.get(v["call_id"])
        res_b = pd.Timestamp(r["rescheduled_berth"]) if r else pred_b
        berth_delay = (pred_b - plan_b).total_seconds() / 3600
        work_delay = max(0.0, (pred_d - plan_d).total_seconds() / 3600 - max(0.0, berth_delay)) if plan_d is not None else 0.0
        queue_wait = max(0.0, (res_b - pred_b).total_seconds() / 3600)
        late = max(0.0, berth_delay)
        box = v["box"]
        cls = L.vessel_class(v["gross_tonnage"])
        charter_h = L.carrier_krw_per_hour(cls)["total_krw_h"]
        bph = L.CLASS_STATS[cls]["box_per_h"]
        extra_night = max(0.0, _night_h(pred_b, pred_d) - (_night_h(plan_b, plan_d) if plan_d is not None else 0.0))
        loss = {"vessel_wait": round((work_delay + queue_wait) * charter_h),       # 선사: 더 머무는 시간 + 앞 배 때문에 기다리는 시간
                "night_surcharge": L.night_surcharge_krw(extra_night * bph),      # 선사·화주: 야간으로 밀린 작업 하역료 50%
                "labor_idle_gang_h": round(late * C, 1),                           # 터미널: 작업조·시간 (금액 미확인)
                "overtime_gang_h": round(work_delay * C, 1)}
        terminal_loss = 0  # 터미널 인건비 단가 미확인 → 금액 합계에서 제외

        acts = []
        if late >= GRADE_H:
            call = pred_b - pd.Timedelta(minutes=30)
            acts.append({"who": "터미널 운영사", "what": f"작업조 {C}개 투입을 {call:%m-%d %H:%M}로 늦추기 (예정 {plan_b:%H:%M} → 예측 {pred_b:%H:%M})",
                         "saving_krw": 0, "saving_gang_h": loss["labor_idle_gang_h"], "kind": "인력 늦추기"})
        if berth_delay <= -GRADE_H:
            acts.append({"who": "터미널 운영사", "what": f"{-berth_delay * 60:.0f}분 일찍 도착 예상 → 선석이 비어 있으면 {pred_b:%H:%M} 착수 준비, 작업조 미리 호출",
                         "saving_krw": round(-berth_delay * charter_h), "kind": "앞당기기"})
        if work_delay >= 2:
            new_work = work_delay * C / (C + 1)
            acts.append({"who": "터미널 운영사", "what": f"작업이 예정 출항보다 {work_delay:.1f}시간 길어질 예상 → 크레인 1기 추가 배정 검토 (출항 예측 평균 오차 약 3시간, 참고용)",
                         "saving_krw": round((work_delay - new_work) * charter_h), "kind": "크레인 추가", "reference_only": True})
        if queue_wait > 0:
            acts.append({"who": "터미널 운영사·선사", "what": f"같은 선석 앞 배 때문에 {queue_wait:.1f}시간 대기 예상 ({r['change_reason']}) → 다른 선석 검토하거나 선사에 감속 입항 안내",
                         "saving_krw": round(queue_wait * charter_h), "kind": "선석 조정"})
        if abs(berth_delay) >= 1:
            acts.append({"who": "선사·대리점", "what": f"도선 신청({RULES['pilot_request_lead_h']['value']}시간 전)·예선 신청({RULES['tug_request_lead_h']['value']}시간 전) 시각을 예측 접안 {pred_b:%m-%d %H:%M} 기준으로 다시 맞추기",
                         "saving_krw": 0, "kind": "도선·예선"})
        if extra_night >= 0.5:
            acts.append({"who": "터미널 운영사", "what": f"작업 {extra_night:.1f}시간이 22~06시로 밀릴 예상 → 야간 작업조 편성 확인 (하역료·인건비 50% 가산)",
                         "saving_krw": 0, "kind": "야간"})
        for a in acts:
            actions.append({**a, "call_id": v["call_id"], "terminal": v["terminal"], "vessel": v["vessel"]})
        vessels.append({"call_id": v["call_id"], "terminal": v["terminal"], "vessel": v["vessel"], "berth_no": v["berth_no"],
                        "berth_source": src, "terminal_planned_berth": v["terminal_planned_berth"], "predicted_berth": f"{pred_b:%Y-%m-%d %H:%M}",
                        "berth_delay_h": round(berth_delay, 2), "work_delay_h": round(work_delay, 2), "queue_wait_h": round(queue_wait, 2),
                        "box": box, "unload": v["unload"], "load": v["load"], "shift": v["shift"],
                        "transship_est": round(box * RULES["transship_share_busan"]["value"]),
                        "gross_tonnage": v["gross_tonnage"], "vessel_class": cls, "carrier_krw_per_h": charter_h,
                        "loss_krw": loss, "terminal_loss_krw": terminal_loss,
                        "carrier_loss_krw": loss["vessel_wait"] + loss["night_surcharge"],
                        "actions": acts})
    vessels.sort(key=lambda x: -(x["terminal_loss_krw"] + x["carrier_loss_krw"]))
    actions.sort(key=lambda a: -a["saving_krw"])

    # 시간대별 필요 작업조: 터미널 예정 vs 예측(일정 조율 반영). 지금 접안 중인 배는 양쪽 다 출항 예정까지
    parts, t0_, up, rec, _ = api._rows(as_of, horizon_h)
    working = [o for o in api._occupancy(parts, t0_, up, rec) if o["state"] == "접안 중"]
    hours = pd.date_range(t0.floor("h"), t0.floor("h") + pd.Timedelta(hours=horizon_h), freq="h", inclusive="left")
    staffing = []
    terms = sorted({v["terminal"] for v in nd["vessels"]} | {o["terminal"] for o in working})
    for term in terms:
        if terminal != "ALL" and term != terminal:
            continue
        plan_iv, pred_iv = [], []
        for v in nd["vessels"]:
            if v["terminal"] != term or not v["terminal_planned_departure"]:
                continue
            plan_iv.append((pd.Timestamp(v["terminal_planned_berth"]), pd.Timestamp(v["terminal_planned_departure"])))
            r = rs.get(v["call_id"])
            pred_iv.append((pd.Timestamp(r["rescheduled_berth"]), pd.Timestamp(r["rescheduled_departure"])) if r else
                           (pd.Timestamp(v["predicted_berth"]), pd.Timestamp(v["predicted_departure"])))
        for o in working:
            if o["terminal"] == term:
                plan_iv.append((o["start"], o["end"]))
                pred_iv.append((o["start"], o["end"]))
        rows = []
        for h in hours:
            a = sum(s <= h < e for s, e in plan_iv) * C
            b = sum(s <= h < e for s, e in pred_iv) * C
            rows.append((h, a, b))
        # 같은 차이가 이어지는 시간대를 한 줄로
        blk = None
        for h, a, b in rows + [(None, None, None)]:
            key = None if h is None else (a, b)
            if blk and key == blk["key"]:
                blk["end"] = h + pd.Timedelta(hours=1)
                continue
            if blk and blk["key"][0] != blk["key"][1]:
                d = blk["key"][1] - blk["key"][0]
                staffing.append({"terminal": term, "from": f"{blk['start']:%m-%d %H:%M}", "to": f"{blk['end']:%m-%d %H:%M}",
                                 "planned_gangs": blk["key"][0], "predicted_gangs": blk["key"][1], "delta": d,
                                 "advice": (f"작업조 {-d}개 줄이거나 다른 배·야드 작업으로 돌리기" if d < 0 else f"작업조 {d}개 더 필요 → 미리 호출"),
                                 "hours": round((blk["end"] - blk["start"]).total_seconds() / 3600, 1)})
            blk = None if h is None else {"key": key, "start": h, "end": h + pd.Timedelta(hours=1)}
    for s in staffing:
        if s["delta"] < 0 and s["hours"] >= 2:  # 1시간짜리 출렁임은 예측 오차 범위라 행동으로 안 올림
            actions.append({"who": "터미널 운영사", "what": f"{s['terminal']} {s['from']}~{s['to'][6:]} 작업조 {-s['delta']}개 여유 → 줄이거나 다른 작업에 배치",
                            "saving_krw": 0, "saving_gang_h": round(-s["delta"] * s["hours"], 1), "kind": "인력 줄이기", "terminal": s["terminal"]})
    actions.sort(key=lambda a: (-a["saving_krw"], -a.get("saving_gang_h", 0)))

    return {"as_of": nd["as_of"], "horizon_h": horizon_h, "terminal": terminal, "model_version": nd["model_version"],
            "method": "규칙 기반 v0 (학습 모델 아님). 금액은 근거 확인된 단가만 (용선료·연료·하역료 할증). 터미널 인건비는 작업조·시간으로만",
            "totals": {"carrier_loss_krw": sum(v["carrier_loss_krw"] for v in vessels),
                       "vessel_wait_krw": sum(v["loss_krw"]["vessel_wait"] for v in vessels),
                       "night_surcharge_krw": sum(v["loss_krw"]["night_surcharge"] for v in vessels),
                       "labor_idle_gang_h": round(sum(v["loss_krw"]["labor_idle_gang_h"] for v in vessels), 1),
                       "possible_saving_krw": sum(a["saving_krw"] for a in actions),
                       "possible_saving_gang_h": round(sum(a.get("saving_gang_h", 0) for a in actions), 1)},
            "assumptions": _assumptions(p), "vessels": vessels, "staffing": staffing, "actions": actions,
            "notes": ["적재 종류: 공개 스케줄에는 양하·선적·선내 이동 개수만 있음. 환적은 부산항 평균 56.7%로 추정, 냉동·위험물 수는 비공개",
                      "작업조 대기는 작업조를 터미널 예정 시각에 맞춰 불렀다고 보고 센 작업조·시간. 실제 호출 방식에 따라 0일 수 있음",
                      "출항(작업) 예측 평균 오차가 약 3시간이라 작업 지연·크레인 추가는 참고용",
                      "다음 배 대기는 같은 선석 번호 기준. HJNC 1B~4B는 물리 선석이 아니라 확인 신호"]}


api.app.router.routes.insert(0, APIRoute("/v25/impact", impact, methods=["GET"]))
