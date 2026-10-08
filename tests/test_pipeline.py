"""
v1.5 기본 테스트: 미래 정보 누출, 스케일러 고정, 드리프트·게이트 규칙.
실행: python -m pytest -q tests
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from data.features import (GROUP_KEY, SEQ_LEN, ArrivalScaler, _history_for, berthed_history, load_samples,
                           split_by_period, upcoming_requests)
from data.hjnc import read_berth_excel

DATA = "data/real/hjnc_berth_schedule_20261008014718.xlsx"


@pytest.fixture(scope="module")
def df():
    return read_berth_excel(DATA)


@pytest.fixture(scope="module")
def samples():
    return load_samples(DATA)


def test_history_never_sees_the_future(df, samples):
    """예측 시점(반입 마감) 뒤에 접안한 배는 이력에 들어가면 안 된다. 자기 자신도 안 된다."""
    hist = berthed_history(df)
    for s in samples[::25]:
        h = _history_for(hist, s["route"], s["decision_time"])
        assert len(h) == SEQ_LEN
        assert (h["입항일시"] <= s["decision_time"]).all()
        assert s["call_id"] not in set(h["모선항차"])


def test_decision_time_is_10h_before_plan(samples):
    for s in samples[::50]:
        assert s["planned_berth"] - s["decision_time"] == pd.Timedelta(hours=10)
        assert s["actual_berth"] > s["decision_time"]


def test_period_split_is_chronological(samples):
    tr, va, te = split_by_period(samples)
    assert max(s["decision_time"] for s in tr) < min(s["decision_time"] for s in va)
    assert max(s["decision_time"] for s in va) < min(s["decision_time"] for s in te)
    assert (len(tr), len(va), len(te)) == (970, 349, 430)


def test_upcoming_only_not_yet_berthed(df):
    t0 = pd.Timestamp("2026-10-08 01:47")
    reqs = upcoming_requests(DATA, t0, 72)
    assert reqs, "10/8~10/10 예정 배가 있어야 함"
    for r in reqs:
        assert pd.isna(r["row"]["작업 시작일시"]) and r["row"]["입항일시"] > t0
        assert len(r["sequence"]) == SEQ_LEN


def test_scaler_is_fixed_and_roundtrips():
    sc = ArrivalScaler()
    before = (sc.delay, sc.box)
    sc.fit([{"actual_correction": 99.0}])           # 데이터로 맞추지 않는다
    assert (sc.delay, sc.box) == before
    for v in (-3.0, 0.0, 0.25, 5.5):
        assert abs(sc.inverse_target(sc.scale_target(v)) - v) < 1e-6


def _window(pred, actual, n=21):
    return [{"predicted": pred, "actual": actual, "baseline": 0.0}] * n


def test_drift_rules():
    from serving_app.monitoring.drift_detector import drift_reasons
    assert drift_reasons(_window(0.2, 0.2)[:20]) == {"relative": False, "absolute": False}   # 21척 전엔 판단 안 함
    assert drift_reasons(_window(0.2, 0.25)) == {"relative": False, "absolute": False}       # 모델이 더 좋음
    assert drift_reasons(_window(0.0, 2.0))["absolute"] is True                              # 오차 2h > 1h
    assert drift_reasons(_window(1.0, 0.0))["relative"] is True                              # 모델이 예정보다 1h 나쁨


def test_gates():
    from serving_app.train_and_register import _gate, _gate_fine_tune
    assert _gate({"mae": 0.2, "baseline_mae": 0.26})
    assert not _gate({"mae": 0.3, "baseline_mae": 0.26})          # 터미널 예정을 못 이기면 배포 안 함
    assert _gate_fine_tune({"mae": 0.2, "prod_mae": 0.25})
    assert not _gate_fine_tune({"mae": 0.3, "prod_mae": 0.25})    # 현재 Production보다 나쁘면 차단
    assert not _gate_fine_tune({"mae": 1.2, "prod_mae": 1.5})     # 절대 기준 1.0h


def test_fallback_prefers_route_mean_when_it_beats_terminal():
    from serving_app.monitoring.retrain_trigger import _choose_fallback
    good_mean = [{"predicted": 1.0, "actual": 0.3, "baseline": 0.0, "mean20": 0.3}] * 21
    bad_mean = [{"predicted": 1.0, "actual": 0.0, "baseline": 0.0, "mean20": 0.5}] * 21
    assert _choose_fallback(good_mean) == "mean20"
    assert _choose_fallback(bad_mean) == "terminal"
    assert _choose_fallback([]) == "terminal"
