# 부산신항 접안 지연 조기경보 — 모델 서빙 & AIOps

SKALA 「모델 서빙 및 AIOps 구성」 조별과제 · 광주 4반 5조 · 최종본 v2.6 (2026-10-08)

교수님 HAIC 실습 구조(LSTM, FastAPI, MLflow 게이트, 드리프트 감지, fine-tune 재학습)를 부산신항 선석배정 실데이터로 바꿨다.
접안 10시간 전(반입 마감) 시점에 그때까지 접안한 배만 보고, 터미널 예정 접안 시각이 얼마나 틀릴지 예측한다.

## 결과 (운영 구간 2026-05~10, 반입 마감 시점 예측 vs 실제 접안)

| 터미널 | 척수 | 터미널 예정 오차 | 모델 오차 | 30분 안 (예정 → 모델) |
|---|---|---|---|---|
| HJNC (학습 터미널) | 434 | 16.5분 | 12.8분 | 85% → 93% |
| PNIT (학습에 안 쓴 터미널) | 289 | 33.9분 | 26.1분 | 72% → 81% |
| BNCT (학습에 안 쓴 터미널, 7~9월) | 190 | 30.4분 | 15.9분 | 57% → 87% |

- 모델은 HJNC 2025년 기록으로만 학습했다. HPNT·DGT는 반입 마감이 실제 접안에 맞춰 다시 계산된 값이라 정답을 만들 수 없어 척수에만 쓴다
- 같은 항로 최근 20척 단순 평균과 비교하면 LSTM의 추가 이득은 작다 (HJNC 13.1분 → 12.8분, 8월 혼잡기 14.5분 → 12.9분)

## 실행

```bash
docker compose up -d --build
```

- 이미지 빌드 안에서 학습 → MLflow 등록(게이트 통과 시 Production)까지 끝난다
- 사용자 화면 http://localhost:8099/early-warning · 관리자 콘솔 http://localhost:8099/admin · API 문서 http://localhost:8099/docs

Docker 없이 실행 (파이썬 3.11)

```bash
pip install -r requirements.txt -r requirements-dev.txt
mkdir -p data/uploads && cp data/real/hjnc_berth_schedule_20261008014718.xlsx data/uploads/
python scripts/train_baseline_v1.py
python serving_app/train_and_register.py
cp data/real/sinhang5_berth_schedule.csv data/uploads/
MODEL_SOURCE=mlflow LOADING_MODE=eager uvicorn addons.dashboard:app --port 8099
python -m pytest -q tests      # 16 passed
```

## API (Swagger에 보이는 10개)

| 구분 | 경로 | 하는 일 |
|---|---|---|
| 서빙 | `GET /health` | 모델 로드·Lazy/Eager·폴백·재학습 중 여부 |
| 서빙 | `POST /predict` | 배 1척 접안 지연 예측 (같은 항로 최근 20척 → LSTM) |
| 서빙 | `GET /predict/next-day` | 다음 48시간 접안 예정 배의 예측 접안·하역시간·출항 |
| AIOps | `POST /predict/batch-test` | 정답 확정 배 → 예측 → 최근 14일 드리프트 판정 → 비동기 재학습 |
| AIOps | `GET /retrain/status` | 재학습 결과, Production 버전·승격 시각 |
| AIOps | `GET /logs`, `GET /logs/{filename}` | aiops.log (감지 → 재학습 → 승격) |
| 데이터 | `POST /data/upload`, `GET /data/status` | 선석 스케줄 업로드(HJNC 엑셀 또는 터미널 통합 CSV), 현황 |
| 활용 | `GET /impact` | 배별 예상 손실·추천 행동, 지연 시간별 손실 기준표 |

화면이 쓰는 내부 경로: `/v25/board`, `/v25/impact`, `/v25/loss-table`, `/v25/lead-accuracy`, `/v25/ais-eta`, `/schedule/*`, `/metrics/*`, `/admin/api/*`

## 운영 규칙

- 배포 게이트: 입력 스키마(422) → 처음 배포는 검증 구간에서 터미널 예정보다 좋아야 함 → 재학습 모델은 운영 모델보다 나쁘지 않아야 함 → MAE 60분 이하. 자동 승격은 운영 모델보다 1분 이상 좋을 때, 아니면 후보로만 등록
- 드리프트: 터미널별 최근 14일·15척 이상에서 모델 MAE ≥ 터미널 예정 MAE + 3분, 또는 RMSE > 60분, 또는 평균 치우침 ≥ 20분
- 재학습: 최근 28일로 fine-tune → 같은 게이트 → 승격 / 실패 시 예전 버전 복원 → 같은 항로 20척 평균 → 터미널 예정 순으로 폴백
- 업무 규칙 값과 출처는 `addons/rules.py` 한 곳에 (확인 안 된 값은 confirmed=False, 계산에 안 씀)

## 폴더

```
serving_app/        FastAPI 서빙 (교수님 구조) · MLflow 학습·게이트·등록 · 드리프트 감지·재학습
addons/             dashboard.py(앱 진입점) · 5개 터미널 계산·화면 API · drift.py(14일 규칙) · impact.py·loss.py(손실)
                    leadtime.py(예측 시점별 정확도) · ais_eta.py(AIS 보정) · admin.py(관리자 콘솔) · rules.py(업무 규칙)
                    early_warning.html(사용자 화면) · admin.html(관리자 콘솔) · terminals.py·refresh.py(데이터 수집·통합)
data/               hjnc.py·features.py(샘플·분할·스케일러) · real/(원본·통합 CSV) · snapshots/(시점별 HJNC 엑셀)
scripts/            학습, 백테스트, 드리프트 시연, 예측표, 시점별 예정 이력
collectors/         AIS 수집기 (aisstream.io, 키 필요)
experiments/        설계 비교·일반화 검증 실험
tests/              pytest 16개 (미래 정보 누출 검사 포함)
```

## 데이터 출처

- 선석배정현황: HJNC·PNIT·HPNT 홈페이지 엑셀, BNCT·DGT 홈페이지 조회 결과 (2026-10-08 수집)
- 선박 제원: 부산항만공사 선박 입출항 집계 2019~2024 (공공데이터포털 15150637), 배 이름으로 연결
- 앞 항만 입항 척수: IMF PortWatch
- 손실 단가: HARPEX 용선료, 항만하역요금표(2023 인가), 싱가포르 LSMGO 시세, 환율 (값과 출처는 `addons/rules.py`)

## 한계

- 48·24시간 전 시점의 터미널 예정은 기록이 남지 않아(최종값으로 덮어씀) 그 시점 정확도는 반입 마감 시점 예정 기준이다. `data/snapshots/` 가 쌓이면 바로잡는다
- AIS 보정은 코드와 연결까지 완성. 무료 수신망(aisstream)이 부산 근해를 덮지 않아 실데이터는 없다
- 출항(작업 종료) 예측은 평균 약 3시간 오차라 작업 지연 금액과 크레인 추가 권고는 참고용
- 손실액 중 초대형선 용선료·묘박 연료량은 가정값
