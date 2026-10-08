"""
선석 스케줄 업로드 (HJNC 정보서비스 엑셀 그대로 또는 같은 컬럼 CSV).
업로드된 파일은 data/uploads/에 쌓이고, 학습·재학습·리플레이·다음날 예측은 항상 가장 최근 파일을 쓴다.
"""
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import SEQ_LEN, load_samples
from data.hjnc import read_berth_excel
from data.storage import UPLOAD_DIR, latest_upload

router = APIRouter(prefix="/data")
REQUIRED_COLUMNS = {"선석", "항로", "모선항차", "반입 마감일시", "입항일시", "출항일시", "작업 시작일시", "작업 완료일시",
                    "양하", "선적", "S/H"}


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    ext = ".xlsx" if name.endswith(".xlsx") else ".csv" if name.endswith(".csv") else None
    if not ext:
        raise HTTPException(400, "HJNC 선석 스케줄 엑셀(.xlsx) 또는 CSV만 업로드할 수 있습니다.")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"berth_{int(time.time())}{ext}")
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        df = read_berth_excel(dest)
    except Exception as e:
        os.remove(dest)
        raise HTTPException(400, f"파일을 읽을 수 없습니다: {e}")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing or len(df) < SEQ_LEN:
        os.remove(dest)
        raise HTTPException(400, f"필수 컬럼 누락 {sorted(missing)} 또는 행 수 부족(최소 {SEQ_LEN})")
    return {"filename": os.path.basename(dest), "rows": int(len(df))}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}
    df = read_berth_excel(path)
    samples = load_samples(path)
    corr = [s["actual_correction"] for s in samples]
    return {"exists": True, "filename": os.path.basename(path), "rows": int(len(df)),
            "completed_calls": int(df["done"].sum()), "upcoming_calls": int((~df["done"]).sum()),
            "labeled_samples": len(samples),
            "first_berth": str(df["입항일시"].min()), "last_berth": str(df["입항일시"].max()),
            "terminal_plan_mae_hours": round(sum(abs(v) for v in corr) / len(corr), 3) if corr else None}
