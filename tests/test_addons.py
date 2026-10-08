"""애드온(선석 체류시간 분해, 수용 능력) 기본 테스트. 기존 모델 테스트와 별개."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addons.berth_time import breakdown, summaries
from addons.capacity import hourly_load
from data.hjnc import read_berth_excel

DATA = "data/real/hjnc_berth_schedule_20261008014718.xlsx"


def test_breakdown_parts_add_up():
    d = breakdown(read_berth_excel(DATA))
    parts = d["wait_start_h"] + d["work_h2"] + d["wait_depart_h"]
    assert ((parts - d["dwell_h"]).abs() < 0.02).mean() > 0.95      # 체류 = 대기 + 작업 + 대기 (음수 보정분 제외)
    s = summaries(d)["headline"]
    assert abs(s["체류 중 작업시작대기 비중(%)"] + s["체류 중 작업 비중(%)"] + s["체류 중 출항대기 비중(%)"] - 100) < 0.5


def test_hourly_load_counts():
    t = pd.Timestamp("2026-10-08 00:00")
    v = pd.DataFrame({"예측접안": [t], "예측작업시작": [t + pd.Timedelta(hours=1)], "예측작업완료": [t + pd.Timedelta(hours=11)],
                      "예측출항": [t + pd.Timedelta(hours=12)], "늦을때_작업완료": [t + pd.Timedelta(hours=14)],
                      "늦을때_출항": [t + pd.Timedelta(hours=15)], "물량": [1000]})
    L = hourly_load(v, t, t + pd.Timedelta(hours=16), qc_rate=25)
    assert L["차있는선석"].sum() == 12 and L["작업중인배"].sum() == 10
    assert L.loc[L["시간당처리_box"] > 0, "시간당처리_box"].iloc[0] == 100 and L["필요QC"].max() == 4
