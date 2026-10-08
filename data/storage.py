"""
업로드된 선석 스케줄 파일 관리. data/uploads/에 타임스탬프 이름으로 쌓이고, 가장 최근 파일 하나를 쓴다.
HJNC 정보서비스에서 내려받은 엑셀(.xlsx) 그대로 또는 같은 컬럼의 CSV.
"""
import glob
import os

UPLOAD_DIR = "data/uploads"


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    files = [f for ext in ("*.xlsx", "*.csv") for f in glob.glob(os.path.join(upload_dir, ext))]
    files.sort(key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            "업로드된 선석 스케줄이 없습니다. 대시보드에서 HJNC 선석 스케줄 엑셀을 먼저 업로드하세요 "
            f"(예시: data/real/hjnc_berth_schedule_20261008014718.xlsx -> {upload_dir}/)."
        )
    return files[-1]
