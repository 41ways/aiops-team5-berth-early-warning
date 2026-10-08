# 부산신항 접안 지연 조기경보 (AIOps 조별과제 광주 4반 5조) — 단일 컨테이너
# 빌드·실행 (이 폴더에서): docker compose up -d --build
#   사용자 화면 http://localhost:8099/early-warning · 관리자 콘솔 /admin · API 문서 /docs
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving_app/ serving_app/
COPY data/ data/
COPY scripts/ scripts/

# 빌드 안에서 학습 → MLflow 등록(게이트 통과 시 Production). 학습은 HJNC 2025년 기록만 쓴다.
# 그 뒤 신항 5개 터미널 통합 CSV를 최신 업로드로 둬서 화면·검증·드리프트 시연이 5개 터미널로 돈다
ENV MLFLOW_DISABLE_AGENT_HINT=1
RUN mkdir -p data/uploads \
 && cp data/real/hjnc_berth_schedule_20261008014718.xlsx data/uploads/build_seed.xlsx \
 && python scripts/train_baseline_v1.py \
 && python serving_app/train_and_register.py \
 && cp data/real/sinhang5_berth_schedule.csv data/uploads/sinhang5_berth_schedule.csv

# addons/ 는 학습 뒤에 복사 → 화면·API만 바꿨을 때 학습 단계를 다시 돌지 않는다
COPY addons/ addons/
COPY collectors/ collectors/

EXPOSE 8099
ENV MODEL_SOURCE=mlflow
ENV LOADING_MODE=eager
ENV TZ=Asia/Seoul

CMD ["uvicorn", "addons.dashboard:app", "--host", "0.0.0.0", "--port", "8099"]
