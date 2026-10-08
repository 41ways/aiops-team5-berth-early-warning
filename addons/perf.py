"""
v2.5: 첫 요청이 수십~100초 걸리던 문제 대응 (2026-10-08 발표 전).

원인: 업로드·재시작 직후 화면이 /v25/board·/predict/next-day·/v25/impact 를 동시에 부르면
      요청마다 같은 무거운 계산(터미널별 전체 이력 예측, 업로드 파일 읽기)을 따로 해서 서로 CPU를 뺏는다.
대응
  1. 계산 잠금: dashboard 의 _split·_history 를 한 번에 하나만 계산하고, 나머지 요청은 그 결과를 같이 쓴다
  2. /data/status 결과를 파일(경로·수정 시각)별로 기억
  3. 미리 계산: 서버가 켜질 때와 업로드 직후 백그라운드에서 한 번 계산해 둔다
기존 serving_app 파일은 고치지 않는다. routes 보다 먼저 import 되어야 /data/status·/data/upload 바뀐 함수가 쓰인다.
"""
import logging
import os
import threading

from addons import dashboard as dv
from serving_app.main import app
from fastapi import File, UploadFile

from serving_app.routers import data as data_router

_lock = threading.RLock()
_log = logging.getLogger("aiops")


def _locked(fn):
    def wrap(*a, **k):
        with _lock:
            return fn(*a, **k)
    wrap.__wrapped__ = fn
    return wrap


dv._split = _locked(dv._split)
dv._history = _locked(dv._history)

_orig_status = data_router.status
_status_cache: dict = {}


def status():
    from data.storage import latest_upload
    try:
        p = latest_upload()
    except FileNotFoundError:
        return {"exists": False}
    key = (p, os.path.getmtime(p))
    with _lock:
        if _status_cache.get("key") != key:
            _status_cache["key"], _status_cache["val"] = key, _orig_status()
        return _status_cache["val"]


_orig_upload = data_router.upload


async def upload(file: UploadFile = File(...)):
    res = await _orig_upload(file)
    warm_async("upload")
    return res


data_router.status = status
data_router.upload = upload


def warm(reason: str = "startup"):
    try:
        with _lock:
            dv.board()
            status()
        from addons import api
        api.next_day(None, 48.0)
        print(f"[perf] cache warmed ({reason})")  # aiops.log 에는 남기지 않음 (운영 이벤트 아님)
    except Exception as e:  # 업로드 파일이 없거나 모델이 아직이면 다음 요청 때 계산
        print(f"[perf] warm skipped ({reason}): {e}")


def warm_async(reason: str):
    threading.Thread(target=warm, args=(reason,), daemon=True).start()


@app.on_event("startup")
def _warm_on_startup():
    warm_async("startup")
