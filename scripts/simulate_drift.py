"""
드리프트 시뮬레이션 (예제 Day3). 서버가 떠 있어야 한다.

  [1] 정상 구간(2026-05~07) 21척 리플레이           → 보통 status ok
  [2] 혼잡 구간(2026-08~10) 21척 리플레이           → 실제 데이터 그대로
  [3] 인위적 드리프트: 정답에 +shift 시간 지연 주입  → 예제의 '변동성 3배' 자리. 감지 → 재학습 → 승격/폴백 확인
  [4] 다시 실제 데이터                              → +shift를 배운 모델이 이번엔 터미널 예정보다 못함
                                                       → relative 드리프트 → 재학습으로 복귀 (또는 폴백)

실행
  python scripts/simulate_drift.py                    # 1 → 2 → 3(+2시간)
  python scripts/simulate_drift.py --shift 3 --target container
"""
import argparse
import time

import requests

TARGETS = {"local": "http://localhost:8077", "container": "http://localhost:8099"}


def send(base, label, **params):
    r = requests.post(f"{base}/predict/replay", params=params, timeout=900)
    r.raise_for_status()
    j = r.json()
    print(f"[{label}] model_mae={j['model_mae']:.3f}h terminal_plan_mae={j['baseline_mae']:.3f}h drift_check={j['drift_check']}")
    if j["drift_check"].get("status") == "retrain_scheduled":   # v1.1 비동기 재학습: 끝날 때까지 기다렸다 결과 출력
        for _ in range(300):
            time.sleep(2)
            st = requests.get(f"{base}/retrain/status", timeout=10).json()
            if not st["running"]:
                print(f"    retrain done (background): {st['last']}")
                break
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS) + ["both"], default="local")
    ap.add_argument("--shift", type=float, default=2.0, help="인위적 지연 주입량(시간)")
    a = ap.parse_args()
    for name in (TARGETS if a.target == "both" else [a.target]):
        base = TARGETS[name]
        print(f"== {name} ({base})")
        try:
            send(base, "normal", period="normal")
            send(base, "busy(real)", period="busy")
            send(base, f"inject+{a.shift}h", period="busy", inject_shift_h=a.shift)
            send(base, "recover(real again)", period="busy")
        except requests.exceptions.ConnectionError:
            print(f"[skip] {base} 에 연결할 수 없습니다. 서버가 떠 있는지(/health) 확인하세요.")
    print("logs/aiops.log 또는 대시보드 '재학습 로그'에서 [WARN] → [INFO] → [OK] 순서를 확인하세요.")


if __name__ == "__main__":
    main()
