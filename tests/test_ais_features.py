"""
v2.1 AIS 특성 함수 테스트. 실데이터가 없으므로 '배 한 척이 15노트로 부산에 다가오는' 가짜 항적으로
계산이 맞는지만 확인한다 (성능 평가가 아님).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.ais_collector import parse
from data.ais_features import (PORT_REF, anchorage_queue, declared_eta, haversine_nm, match_mmsi, physics_eta,
                               track_sequence)


def fake_track(mmsi=440000001, hours=30, sog=15.0, start_nm=450.0, t0="2026-10-08 00:00"):
    t = pd.date_range(t0, periods=hours * 6, freq="10min")
    dist = start_nm - sog * np.arange(len(t)) / 6
    lat = PORT_REF[0] - dist / 60.0          # 정남쪽에서 북상 (위도 1도 = 60해리)
    return pd.DataFrame({"mmsi": mmsi, "t_kst": t, "lat": lat, "lon": PORT_REF[1], "sog": sog,
                         "dist_nm": haversine_nm(lat, PORT_REF[1], *PORT_REF)})


def test_haversine_one_degree_lat_is_60nm():
    assert abs(haversine_nm(34.0, 128.8, 35.0, 128.8) - 60.0) < 0.2


def test_track_uses_only_past_and_physics_eta():
    pos = fake_track()
    as_of = pd.Timestamp("2026-10-08 20:00")
    seq, q = track_sequence(pos, 440000001, as_of)
    assert seq.shape == (20, 2) and q["coverage"] == 1.0
    assert (pos[pos["t_kst"] <= as_of]["dist_nm"].min() - seq[-1, 0]) < 3     # as_of 이후 위치는 안 씀
    eta = physics_eta(seq, as_of)
    remaining_h = seq[-1, 0] / 15.0
    assert abs((eta - as_of).total_seconds() / 3600 - remaining_h) < 0.01


def test_anchored_vessel_has_no_physics_eta_and_counts_in_queue():
    pos = fake_track(sog=0.2, start_nm=3.0)
    pos["lat"], pos["lon"] = 34.98, 128.83
    as_of = pd.Timestamp("2026-10-08 20:00")
    seq, _ = track_sequence(pos, 440000001, as_of)
    assert physics_eta(seq, as_of) is None
    assert anchorage_queue(pos, as_of) == 1


def test_declared_eta_year_rollover():
    row = {"eta_month": 1, "eta_day": 2, "eta_hour": 3, "eta_minute": 0}
    t = declared_eta(row, pd.Timestamp("2026-12-31 10:00"))
    assert t == pd.Timestamp("2027-01-02 12:00")               # UTC 03:00 → KST 12:00, 다음 해
    assert declared_eta({"eta_month": 0, "eta_day": 0, "eta_hour": 24, "eta_minute": 60}, pd.Timestamp("2026-10-08")) is None


def test_match_mmsi_by_name():
    static = pd.DataFrame({"time_utc": ["2026-10-08T00:00", "2026-10-08T01:00"], "mmsi": [1, 2],
                           "name": ["YONG SHUN", "ONE MILANO "]})
    hjnc = pd.DataFrame({"선박명": ["YONG SHUN", "ONE MILANO", "ULTIMA"]})
    assert match_mmsi(static, hjnc) == {"YONG SHUN": 1, "ONE MILANO": 2}


def test_collector_parse():
    msg = {"MessageType": "PositionReport", "MetaData": {"MMSI": 440000001, "ShipName": "YONG SHUN ", "time_utc": "2026-10-08 00:00:00"},
           "Message": {"PositionReport": {"Latitude": 34.9, "Longitude": 128.8, "Sog": 14.2, "Cog": 10.0,
                                          "TrueHeading": 12, "NavigationalStatus": 0}}}
    kind, row = parse(msg)
    assert kind == "pos" and row[1] == 440000001 and row[2] == "YONG SHUN" and row[5] == 14.2
    msg = {"MessageType": "ShipStaticData", "MetaData": {"MMSI": 440000001},
           "Message": {"ShipStaticData": {"Name": "YONG SHUN", "ImoNumber": 9000000, "CallSign": "ABCD", "Type": 70,
                                          "Eta": {"Month": 10, "Day": 8, "Hour": 19, "Minute": 0},
                                          "MaximumStaticDraught": 9.5, "Destination": "KRPUS", "Dimension": {"A": 100, "B": 40}}}}
    kind, row = parse(msg)
    assert kind == "static" and row[6:10] == [10, 8, 19, 0] and row[11] == "KRPUS" and row[12] == 140
