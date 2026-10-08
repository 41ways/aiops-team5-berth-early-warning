"""
모델 불러오기와 예측. 예제 구조 그대로 (Lazy/Eager, local/mlflow, 스케일러는 항상 로컬 scaler.pkl).

환경변수: LOADING_MODE = lazy(기본) | eager,  MODEL_SOURCE = local(기본) | mlflow
          MODEL_POLL_SEC = 30 (mlflow일 때 Production 버전을 확인하는 주기, 0이면 끔)

v1.1 (교수님 Wrap-Up 2번 "복제본 간 조율·캐시 무효화 없음"):
  서버가 여러 개 떠 있으면 재학습·승격한 서버만 reset_cache()로 새 모델을 쓴다.
  그래서 요청이 올 때 MODEL_POLL_SEC마다 레지스트리의 Production 버전을 확인하고, 바뀌었으면 다시 불러온다.
  (레지스트리가 공유 DB/S3에 있어야 의미가 있다. 로컬 sqlite면 같은 PC의 프로세스끼리만 공유)
"""
import os
import time

from data.features import ArrivalScaler

LOCAL_MODEL_PATH = "serving_app/models/arrival_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
MODEL_NAME = "BERTH_ARRIVAL_Corrector"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/Production"

_model_cache = None
_last_check = 0.0


class LoadedModel:
    def __init__(self, keras_model, scaler: ArrivalScaler, version: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version

    def predict_one(self, sequence: list[dict]) -> float:
        """같은 항로 최근 20척 → 다음 배의 접안 오차(h). 흐름: 0~1 변환(빈칸2) → (1,20,2) → 예측 → 시간 복원(빈칸3)"""
        return self.predict_many([sequence])[0]

    def predict_many(self, sequences: list[list[dict]]) -> list[float]:
        import numpy as np

        x = np.array([[self.scaler.transform_point(p["delay_hours"], p["box"]) for p in s] for s in sequences],
                     dtype="float32")
        pred_scaled = self._keras_model.predict(x, verbose=0).flatten()
        return [self.scaler.inverse_target(float(v)) for v in pred_scaled]


def _load_from_local() -> LoadedModel:
    from tensorflow import keras

    return LoadedModel(keras.models.load_model(LOCAL_MODEL_PATH), ArrivalScaler.load(SCALER_PATH), "v1-local")


def production_version() -> str | None:
    try:
        from mlflow.tracking import MlflowClient

        vs = [v for v in MlflowClient().search_model_versions(f"name='{MODEL_NAME}'") if v.current_stage == "Production"]
        return max(vs, key=lambda v: int(v.version)).version if vs else None
    except Exception:
        return None


def _load_from_mlflow() -> LoadedModel:
    import mlflow.tensorflow

    v = production_version()
    keras_model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/{v}" if v else MLFLOW_MODEL_URI)
    scaler = ArrivalScaler.load(SCALER_PATH)  # 빈칸 5: 모델은 레지스트리에서, 스케일러는 학습 때 쓴 로컬 파일
    return LoadedModel(keras_model, scaler, f"production:v{v}" if v else "production")


def _load_model() -> LoadedModel:
    return _load_from_mlflow() if os.getenv("MODEL_SOURCE", "local") == "mlflow" else _load_from_local()


def load_eager() -> LoadedModel:
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def _stale() -> bool:
    """다른 서버가 새 버전을 승격했는지 주기적으로 확인."""
    global _last_check
    poll = float(os.getenv("MODEL_POLL_SEC", "30"))
    if _model_cache is None or poll <= 0 or os.getenv("MODEL_SOURCE", "local") != "mlflow":
        return False
    if time.time() - _last_check < poll:
        return False
    _last_check = time.time()
    v = production_version()
    if v and _model_cache.version != f"production:v{v}":
        print(f"[poll] registry Production v{v} != serving {_model_cache.version} -> reload")
        return True
    return False


def get_model() -> LoadedModel:
    global _model_cache
    if _stale():
        _model_cache = None
    if _model_cache is None:  # 빈칸 4: 비어 있을 때만 한 번 불러옴
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache


def reset_cache():
    """새 버전 승격 시 캐시 무효화 (교수님 Wrap-Up 2번: 예제는 승격 후에도 캐시가 그대로 남음)."""
    global _model_cache
    _model_cache = None
