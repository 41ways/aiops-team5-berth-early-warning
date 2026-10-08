"""
v2.5: 최신 선석 스케줄 다시 받기 → 5개 터미널 통합 CSV 다시 만들기 → 서버에 업로드.

  1. HJNC 엑셀 (홈페이지 엑셀 버튼과 같은 요청, 2025-01-01 ~ 내년 말) → data/real/hjnc_berth_schedule_<내보낸 시각>.xlsx
  2. PNIT·HPNT 이번 분기 (홈페이지 엑셀과 같은 요청) → data/real/terminals_raw/ 덮어쓰기
     BNCT는 robots.txt가 수집을 막아 자동으로 받지 않는다 (직접 엑셀 버튼으로). DGT도 이번엔 받지 않는다
  3. addons/terminals.py 로 data/real/sinhang5_berth_schedule.csv 다시 만들기
  4. POST {서버}/data/upload  (모델은 다시 학습하지 않는다. 대시보드·검증만 새 데이터로)

실행 (app/ 에서): python addons/refresh.py [--server http://localhost:8099]
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import date

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UA = {"User-Agent": "Mozilla/5.0 (SKALA class project; manual-equivalent export)"}


def hjnc():
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://www.hjnc.co.kr/esvc/vessel/berthScheduleT", timeout=30)
    body = {"startDate": "2025-01-01", "endDate": f"{date.today().year + 1}-12-31", "sort": "ETB", "dateType": None}
    s.get("https://www.hjnc.co.kr/esvc/vessel/berthScheduleT/countExcelRows", params={k: v or "" for k, v in body.items()}, timeout=30)
    r = s.post("https://www.hjnc.co.kr/esvc/vessel/berthScheduleT/download", json=body, timeout=180,
               headers={"Referer": "https://www.hjnc.co.kr/esvc/vessel/berthScheduleT", "X-Requested-With": "XMLHttpRequest"})
    r.raise_for_status()
    if not r.content:
        raise RuntimeError("HJNC 응답이 비었습니다")
    m = re.search(r"(\d{14})", r.headers.get("Content-Disposition", ""))
    out = f"data/real/hjnc_berth_schedule_{m.group(1) if m else 'latest'}.xlsx"
    open(out, "wb").write(r.content)
    return out


def pnit_like(host, code, name):
    s = requests.Session()
    s.headers.update(UA)
    s.get(f"https://{host}/infoservice/vessel/vslScheduleList.jsp", timeout=30)
    y, q = date.today().year, (date.today().month - 1) // 3
    start = f"{y}-{3 * q + 1:02d}-01"
    end = f"{y}-{3 * q + 3:02d}-{30 if 3 * q + 3 in (6, 9, 11) else 31}"
    r = s.get(f"https://{host}/infoservice/vessel/vslScheduleList_excel.jsp", timeout=60,
              params={"isSearch": "Y", "page": 1, "groupID": "U999", "tmnCod": code, "strdStDate": start, "strdEdDate": "2026-11-15" if end < "2026-11-15" else end, "route": ""})
    r.raise_for_status()
    old = [f for f in os.listdir("data/real/terminals_raw") if f"_{name}_{start}" in f]
    for f in old:
        os.remove(os.path.join("data/real/terminals_raw", f))
    out = f"data/real/terminals_raw/{date.today():%Y%m%d}_{name}_{start}_{end}.xls"
    open(out, "wb").write(r.content)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8099")
    a = ap.parse_args()
    print("HJNC", hjnc())
    for host, code, name in (("www.pnitl.com", "P", "pnit"), ("www.hpnt.co.kr", "H", "hpnt")):
        try:
            print(name.upper(), pnit_like(host, code, name))
        except Exception as e:
            print(name.upper(), "실패, 이전 파일 유지:", e)
    subprocess.run([sys.executable, "addons/terminals.py"], check=True)
    with open("data/real/sinhang5_berth_schedule.csv", "rb") as f:
        r = requests.post(f"{a.server}/data/upload", files={"file": ("sinhang5_v2.5.csv", f, "text/csv")}, timeout=120)
    print("업로드", r.status_code, r.text[:200])
