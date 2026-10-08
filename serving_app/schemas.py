"""
요청/응답 스키마. 학습 피처(data/features.py)와 같은 길이·범위를 강제한다.
접안 오차는 음수(예정보다 일찍 접안)가 나올 수 있으므로 예제의 gt=0 제약을 두지 않는다.
"""
from pydantic import BaseModel, Field

from data.features import SEQ_LEN


class HistoryPoint(BaseModel):
    delay_hours: float = Field(..., ge=-48, le=48, description="그 배의 접안 오차(h) = 실제 접안 − 터미널 접안 예정")
    box: float = Field(..., ge=0, le=30000, description="그 배의 처리 물량(양하+선적+S/H, box)")


class PredictRequest(BaseModel):
    sequence: list[HistoryPoint] = Field(..., min_length=SEQ_LEN, max_length=SEQ_LEN,
                                         description=f"같은 항로 최근 {SEQ_LEN}척 (오래된 배 → 최근 배)")
    planned_berth: str | None = Field(None, description="이 배의 터미널 접안 예정 'YYYY-MM-DD HH:MM' (주면 예측 접안 시각도 돌려줌)")


class PredictResponse(BaseModel):
    predicted_delay_hours: float
    predicted_berth: str | None = None
    model_version: str


class CaseIn(BaseModel):
    sequence: list[HistoryPoint] = Field(..., min_length=SEQ_LEN, max_length=SEQ_LEN)
    actual_delay_hours: float = Field(..., ge=-48, le=48, description="정답: 실제 접안 − 터미널 예정")
    decision_time: str | None = None


class BatchTestRequest(BaseModel):
    cases: list[CaseIn] = Field(..., min_length=1)


class BatchTestResponse(BaseModel):
    predictions: list[float]
    actuals: list[float]
    model_mae: float
    baseline_mae: float
    drift_check: dict
