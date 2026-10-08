"""
AIS → 모델 입력 (v2.1, 확장 시제품. 실데이터가 쌓이면 바로 쓰도록 만든 틀).

핵심 아이디어: 교수님 파이프라인의 (20, 2) 모양을 그대로 쓴다.
  지금 모델 A : 같은 항로 최근 20척 × [접안 오차, 물량]       → 접안 오차   (항로의 '습관')
  확장 모델 B : 그 배의 최근 20시간 × [부산까지 거리(해리), 속력] → 접안 오차   (그 배의 '지금 상태')
  최종 = 터미널 예정 + 가중합(A, B). 가중치는 리드타임(72/48/24/10시간 전)별로 검증 구간에서 정한다.

좌표·반경은 대략값이다 (부산신항 입구 부근). 실제 도선점·묘박지 좌표로 바꿔 쓸 것.
"""
import math
import re

import numpy as np
import pandas as pd

PORT_REF = (35.03, 128.80)        # 부산신항 입구 부근 (대략값, 위도·경도)
ANCHORAGE = ((34.98, 128.83), 8)  # 신항 외곽 묘박지 부근 중심, 반경(해리) (대략값)
SEQ_HOURS = 20
NM_PER_KM = 1 / 1.852


def haversine_nm(lat1, lon1, lat2, lon2):
    """두 지점 사이 거리(해리). 배열도 됨."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a)) * NM_PER_KM


def load_positions(paths) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["time_utc"] = pd.to_datetime(df["time_utc"].astype(str).str.slice(0, 19), utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc", "lat", "lon"])
    df["t_kst"] = df["time_utc"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    df["dist_nm"] = haversine_nm(df["lat"], df["lon"], *PORT_REF)
    return df.sort_values(["mmsi", "t_kst"]).reset_index(drop=True)


def _norm_name(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def match_mmsi(static_df: pd.DataFrame, hjnc_df: pd.DataFrame) -> dict:
    """HJNC 선박명 ↔ AIS Name 정확 일치(공백·기호 무시). 같은 이름이 여러 MMSI면 최근 것을 쓴다."""
    s = static_df.dropna(subset=["name"]).copy()
    s["key"] = s["name"].map(_norm_name)
    last = s.sort_values("time_utc").groupby("key")["mmsi"].last()
    return {n: int(last[_norm_name(n)]) for n in hjnc_df["선박명"].dropna().unique() if _norm_name(n) in last.index}


def track_sequence(pos: pd.DataFrame, mmsi: int, as_of: pd.Timestamp, hours: int = SEQ_HOURS):
    """as_of 직전 hours시간을 1시간 간격으로: [부산까지 거리(해리), 속력(노트)]. 빈 시간은 앞 값으로 채움.
    반환: (array (hours, 2) 또는 None, 품질 dict). as_of 이후 신호는 절대 쓰지 않는다."""
    g = pos[(pos["mmsi"] == mmsi) & (pos["t_kst"] <= as_of) & (pos["t_kst"] > as_of - pd.Timedelta(hours=hours))]
    if g.empty:
        return None, {"coverage": 0.0, "last_seen_h": None}
    h = g.set_index("t_kst")[["dist_nm", "sog"]].resample("1h").last()
    idx = pd.date_range(as_of.floor("h") - pd.Timedelta(hours=hours - 1), as_of.floor("h"), freq="h")
    h = h.reindex(idx)
    coverage = float(h["dist_nm"].notna().mean())
    h = h.ffill().bfill()
    last_seen = (as_of - g["t_kst"].max()).total_seconds() / 3600
    return h.values.astype("float32"), {"coverage": round(coverage, 2), "last_seen_h": round(last_seen, 1)}


def physics_eta(seq: np.ndarray, as_of: pd.Timestamp, min_sog: float = 3.0):
    """거리 ÷ 최근 6시간 평균 속력. 정박 중(속력 < 3노트)이면 계산하지 않음."""
    if seq is None:
        return None
    sog = float(np.nanmean(seq[-6:, 1]))
    if not np.isfinite(sog) or sog < min_sog:
        return None
    return as_of + pd.Timedelta(hours=float(seq[-1, 0]) / sog)


def declared_eta(row, as_of: pd.Timestamp):
    """AIS에 선원이 입력한 ETA(월·일·시·분, 연도 없음) → as_of 기준 가장 가까운 미래 시각."""
    try:
        m, d, hh, mm = int(row["eta_month"]), int(row["eta_day"]), int(row["eta_hour"]), int(row["eta_minute"])
        if not (1 <= m <= 12 and 1 <= d <= 31 and hh < 24 and mm < 60):
            return None
        for y in (as_of.year, as_of.year + 1):
            t = pd.Timestamp(year=y, month=m, day=d, hour=hh, minute=mm) + pd.Timedelta(hours=9)  # UTC → KST
            if t >= as_of - pd.Timedelta(days=2):
                return t
    except (ValueError, TypeError):
        return None
    return None


def anchorage_queue(pos: pd.DataFrame, as_of: pd.Timestamp, window_h: float = 2.0) -> int:
    """as_of 직전 window_h시간 동안 묘박지 반경 안에서 거의 멈춰 있던(속력 < 1노트) 배 수 = 대기 줄 길이."""
    (lat, lon), r = ANCHORAGE
    g = pos[(pos["t_kst"] <= as_of) & (pos["t_kst"] > as_of - pd.Timedelta(hours=window_h))]
    g = g[(haversine_nm(g["lat"], g["lon"], lat, lon) <= r) & (g["sog"] < 1.0)]
    return int(g["mmsi"].nunique())


def scalar_features(pos, static_row, mmsi, as_of, planned_berth):
    """모델 B 보조 특성 (예정 대비 시간 차, 시간 단위). 없으면 NaN."""
    seq, q = track_sequence(pos, mmsi, as_of)
    pe = physics_eta(seq, as_of)
    de = declared_eta(static_row, as_of) if static_row is not None else None
    f = lambda t: (t - planned_berth).total_seconds() / 3600 if t is not None else np.nan
    return {"physics_eta_minus_plan_h": f(pe), "declared_eta_minus_plan_h": f(de),
            "anchorage_queue": anchorage_queue(pos, as_of), "ais_coverage": q["coverage"],
            "ais_last_seen_h": q["last_seen_h"]}, seq
