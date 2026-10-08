"""
실시간 AIS 수집기 (v2.1, 확장 시제품). aisstream.io 웹소켓 → 날짜별 CSV.

이 파일은 회사/집 PC에서 돌린다 (이 저장소의 분석 서버는 외부망이 막혀 있을 수 있음).
  1) https://aisstream.io 에서 GitHub 계정으로 로그인 → API 키 발급 (키는 절대 커밋하지 말 것)
  2) pip install -r requirements-collect.txt
  3) AISSTREAM_API_KEY=발급키 python collectors/ais_collector.py
     → data/ais/positions_YYYYMMDD.csv, data/ais/static_YYYYMMDD.csv (UTC 기준)

기본 수집 범위: 동중국해~대한해협 (상하이·닝보에서 부산까지 약 24~36시간 거리).
  더 멀리(72시간 전)는 지상 수신국 범위를 벗어나 빈 구간이 생길 수 있다 → 그 리드타임은 선사 스케줄·Port-MIS로 보완.
메시지 형식: github.com/aisstream/ais-message-models (PositionReport: Sog·Cog·NavigationalStatus·Latitude·Longitude,
            ShipStaticData: Name·ImoNumber·Eta{Month,Day,Hour,Minute}·Destination·MaximumStaticDraught)
서비스는 SLA·재전송이 없다 → 끊기면 자동 재접속하고, 받은 건 바로 파일에 쓴다.
"""
import asyncio
import csv
import json
import zlib
import os
import sys
import time
from datetime import datetime, timezone

URL = "wss://stream.aisstream.io/v0/stream"
# [[남서 위도, 남서 경도], [북동 위도, 북동 경도]] — 동중국해~대한해협
DEFAULT_BOX = [[[29.0, 121.0], [36.5, 131.5]]]
OUT_DIR = os.getenv("AIS_OUT_DIR", "data/ais")
POS_COLS = ["time_utc", "mmsi", "name", "lat", "lon", "sog", "cog", "heading", "nav_status"]
STA_COLS = ["time_utc", "mmsi", "name", "imo", "callsign", "ship_type", "eta_month", "eta_day", "eta_hour",
            "eta_minute", "draught", "destination", "length_m"]


def _writer(kind: str, cols: list[str]):
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(OUT_DIR, f"{kind}_{day}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(cols)
    return path, f, w


def decode(raw):
    """aisstream 메시지 → dict. 구독 확인에 CompressionEnabled=True 가 오면 본문이 압축된 바이트로 온다 → 순서대로 풀어 본다."""
    if isinstance(raw, str):
        return json.loads(raw)
    for fn in (lambda b: b.decode("utf-8"),
               lambda b: zlib.decompress(b, 47).decode("utf-8"),        # zlib/gzip 자동
               lambda b: zlib.decompress(b, -15).decode("utf-8"),       # raw deflate
               lambda b: __import__("zstandard").ZstdDecompressor().decompress(b, max_output_size=2 ** 24).decode("utf-8")):
        try:
            return json.loads(fn(bytes(raw)))
        except Exception:
            continue
    return None


def parse(msg: dict):
    """aisstream 메시지 1개 → ('pos'|'static', row) 또는 None. MetaData 키가 바뀌어도 본문 값으로 대체."""
    meta = msg.get("MetaData", {}) or {}
    kind = msg.get("MessageType")
    body = (msg.get("Message") or {}).get(kind, {}) or {}
    t = meta.get("time_utc") or datetime.now(timezone.utc).isoformat()
    mmsi = meta.get("MMSI") or body.get("UserID")
    name = (meta.get("ShipName") or body.get("Name") or "").strip()
    if kind in ("PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport"):
        return "pos", [t, mmsi, name, body.get("Latitude", meta.get("latitude")), body.get("Longitude", meta.get("longitude")),
                       body.get("Sog"), body.get("Cog"), body.get("TrueHeading"), body.get("NavigationalStatus")]
    if kind == "ShipStaticData":
        eta = body.get("Eta") or {}
        dim = body.get("Dimension") or {}
        return "static", [t, mmsi, name, body.get("ImoNumber"), (body.get("CallSign") or "").strip(), body.get("Type"),
                          eta.get("Month"), eta.get("Day"), eta.get("Hour"), eta.get("Minute"),
                          body.get("MaximumStaticDraught"), (body.get("Destination") or "").strip(),
                          (dim.get("A") or 0) + (dim.get("B") or 0)]
    return None


async def run(api_key: str, boxes, mmsi_filter: list[str] | None):
    import websockets  # pip install websockets

    os.makedirs(OUT_DIR, exist_ok=True)
    sub = {"APIKey": api_key, "BoundingBoxes": boxes,
           "FilterMessageTypes": ["PositionReport", "ShipStaticData", "StandardClassBPositionReport"]}
    if mmsi_filter:
        sub["FiltersShipMMSI"] = mmsi_filter[:200]
    backoff = 1
    while True:
        try:
            async with websockets.connect(URL, ping_interval=20, max_size=2 ** 22) as ws:
                await ws.send(json.dumps(sub))          # 연결 후 3초 안에 보내야 함
                print(f"[ais] connected, box={boxes} (메시지가 오면 1분마다 상태를 출력합니다. 아무 줄도 안 나오면 받은 메시지 0건)")
                backoff = 1
                files = {}
                n, t0 = 0, time.time()
                last_beat, other = time.time(), 0
                async for raw in ws:
                    msg = decode(raw)
                    if msg is None:
                        other += 1
                        if other <= 3:
                            print(f"[ais] 해석 못 한 메시지 (앞 40바이트): {bytes(raw[:40]) if isinstance(raw, (bytes, bytearray)) else raw[:80]}")
                        continue
                    if isinstance(msg, dict) and ("error" in msg or "Error" in msg):   # 키 오류 등은 서버가 이 형태로 알려 준다
                        print(f"[ais] 서버 오류 메시지: {msg}")
                    out = parse(msg)
                    if time.time() - last_beat > 60:              # 1분마다 받은 양을 보여 줘서 '조용히 0건'을 바로 알게
                        print(f"[ais] 최근 상태: 저장 {n}건, 저장 안 한 메시지 {other}건")
                        last_beat = time.time()
                    if not out:
                        other += 1
                        if other <= 3:
                            print(f"[ais] 저장 안 한 메시지 예: {str(msg)[:200]}")
                        continue
                    kind, row = out
                    key = (kind, datetime.now(timezone.utc).strftime("%Y%m%d"))
                    if key not in files:                 # 날짜가 바뀌면 새 파일
                        for k in [k for k in files if k[0] == kind]:
                            files.pop(k)[1].close()
                        files[key] = _writer("positions" if kind == "pos" else "static", POS_COLS if kind == "pos" else STA_COLS)
                    files[key][2].writerow(row)
                    n += 1
                    if n % 2000 == 0:
                        for v in files.values():
                            v[1].flush()
                        print(f"[ais] {n} msgs, {n / (time.time() - t0):.0f}/s")
        except Exception as e:  # 끊기면 재접속 (서비스에 재전송이 없으므로 빨리 붙는 게 중요)
            print(f"[ais] disconnected: {e} -> retry in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def main():
    key = os.getenv("AISSTREAM_API_KEY")
    if not key:
        sys.exit("AISSTREAM_API_KEY 환경변수가 필요합니다 (aisstream.io에서 발급)")
    mmsi = [m for m in os.getenv("AIS_MMSI", "").split(",") if m]
    # AIS_BOX=world 로 주면 전 세계 범위 (키가 맞는지 확인용: 맞으면 몇 초 안에 메시지가 쏟아진다)
    box = [[[-90, -180], [90, 180]]] if os.getenv("AIS_BOX") == "world" else DEFAULT_BOX
    asyncio.run(run(key, box, mmsi or None))


if __name__ == "__main__":
    main()
