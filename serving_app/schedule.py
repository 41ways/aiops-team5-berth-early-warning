"""
일정표용 보조 계산: 작업시간 → 예측 출항(선석 반납) 시각.

실험(experiments/hjnc_route_variant.py) 결과 LSTM보다 단순한 계산이 더 정확해서 이것을 쓴다.
  기본 작업시간 = a + b × 물량            (완료 선박 최소제곱. 백테스트는 2026-04 이전 배로만, 예측표는 전체)
  항로 보정비   = 같은 항로 최근 10척의 평균 log(실제/기본)
  예측 작업시간 = 기본 × exp(항로 보정비)
  예측 출항     = 예측 접안 + 접안→작업시작 중앙값 + 예측 작업시간 + 작업완료→출항 중앙값
테스트(2026-05~10) 작업시간 MAE: 기본 3.51h → 항로 보정 3.18h
v1.4: 트리 부스팅(동시 작업 배 수·선석·요일·양하 비율 등 12개 특성)은 3.04h로 4% 낫지만 복잡도 대비 이득이 작아
      채택하지 않음 (experiments/worktime_gbm.py). 대신 80% 범위(예측 작업시간 × 0.78 ~ × 1.32)를 붙임
"""
import numpy as np
import pandas as pd

from data.hjnc import completed_calls


class WorkTimeEstimator:
    def __init__(self, df: pd.DataFrame, fit_before: pd.Timestamp | None = None):
        d = completed_calls(df)
        tr = d[d["입항일시"] < fit_before] if fit_before is not None else d
        A = np.stack([np.ones(len(tr)), tr["box"].values], 1)
        (self.a, self.b), *_ = np.linalg.lstsq(A, tr["work_h"].values, rcond=None)
        self.pre_h = float(tr["pre_h"].clip(lower=0).median())
        self.post_h = float(tr["post_h"].clip(lower=0).median())
        self.done = d.assign(lr=np.log(d["work_h"] / np.clip(self.a + self.b * d["box"], 1.0, None)))

    def base(self, box: float) -> float:
        return max(self.a + self.b * box, 1.0)

    def work_hours(self, box: float, route: str, as_of: pd.Timestamp) -> float:
        g = self.done[(self.done["항로"] == route) & (self.done["작업 완료일시"] < as_of)]["lr"].tail(10)
        return self.base(box) * float(np.exp(g.mean() if len(g) else 0.0))

    def departure(self, berth_time: pd.Timestamp, box: float, route: str, as_of: pd.Timestamp,
                  factor: float = 1.0) -> pd.Timestamp:
        """factor: 작업시간에 곱하는 배수. 80% 범위는 residual_band.json의 work_x10 / work_x90 (v1.4)."""
        return berth_time + pd.Timedelta(hours=self.pre_h + factor * self.work_hours(box, route, as_of) + self.post_h)
