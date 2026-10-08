"""
MLflow 학습 → 기록 → 게이트 → 등록 → Production 승격 (예제 Day2), 드리프트 후 fine-tuning (예제 Day3).

데이터 분할 (data/features.py)
  학습 2025-01 ~ 2025-12  (이 중 마지막 15%로 조기 종료 판단)
  검증 2026-01 ~ 2026-04  → 배포 게이트
  테스트 2026-05 ~        → scripts/backtest.py, 드리프트 리플레이

배포 게이트 (예제의 RMSE ≤ $4 자리)
  최초 학습: 검증 구간 모델 MAE < 터미널 예정 MAE  그리고  MAE ≤ 1.0시간
  재학습   : 최근 홀드아웃에서 새 모델 MAE ≤ 현재 Production MAE  그리고  MAE ≤ 1.0시간 (챔피언/챌린저)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import ArrivalScaler, build_sequences, load_samples, split_by_period, train_test_split
from data.storage import latest_upload
from serving_app.lstm_model import build_model

# 시드: 1·2·3·42 중 검증 MAE가 가장 낮은 1을 고정 (테스트로 고르지 않음). 시드별 테스트 MAE 0.213~0.231h
SEED = 1
keras.utils.set_random_seed(SEED)

ABS_MAE_GATE = 1.0
MODEL_NAME = "BERTH_ARRIVAL_Corrector"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 100
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    return {"rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "baseline_mae": float(np.mean(np.abs(y_true)))}   # 보정 0 = 터미널 예정 그대로


def _gate(m: dict) -> bool:
    return m["mae"] < m["baseline_mae"] and m["mae"] <= ABS_MAE_GATE


def _gate_fine_tune(m: dict) -> bool:
    """재학습 게이트(챔피언/챌린저): 같은 최근 홀드아웃에서 현재 Production보다 나쁘지 않고 MAE ≤ 1.0h.
    최근 21일 홀드아웃은 9~10척뿐이라 '터미널 예정보다 나음'을 매번 요구하면 우연에 좌우된다.
    터미널 예정과의 비교는 드리프트 감지(relative)와 폴백이 맡는다."""
    return m["mae"] <= m["prod_mae"] and m["mae"] <= ABS_MAE_GATE


def _register_if_gate_passed(run_id: str, m: dict, gate=_gate) -> dict:
    result = {"run_id": run_id, **m, "promoted": False}
    if gate(m):
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(name=MODEL_NAME, version=v.version, stage="Production",
                                                      archive_existing_versions=True)
        result.update(promoted=True, version=v.version)
        print(f"[GATE PASSED] mae={m['mae']:.3f}h (terminal plan {m['baseline_mae']:.3f}h"
              f"{', current prod %.3fh' % m['prod_mae'] if 'prod_mae' in m else ''}) -> {MODEL_NAME} v{v.version} Production")
    else:
        print(f"[GATE FAILED] mae={m['mae']:.3f}h, terminal plan={m['baseline_mae']:.3f}h -> 배포 차단, 기존 Production 유지")
    return result


def fit_base(train: list[dict], scaler: ArrivalScaler):
    keras.utils.set_random_seed(SEED)
    X, y = build_sequences(train, scaler)
    ys = np.array([scaler.scale_target(v) for v in y], dtype="float32")
    model = build_model()
    model.fit(X, ys, epochs=BASE_EPOCHS, batch_size=32, verbose=0, validation_split=0.15,
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    return model


def predict(model, samples: list[dict], scaler: ArrivalScaler) -> list[float]:
    X, _ = build_sequences(samples, scaler)
    return [scaler.inverse_target(float(p)) for p in model.predict(X, verbose=0).flatten()]


def train_and_register(path: str | None = None) -> dict:
    """처음부터(scratch) 학습. 데이터가 충분한 base 학습에서만 사용."""
    samples = load_samples(path or latest_upload())
    train, val, _ = split_by_period(samples)
    scaler = ArrivalScaler.load(SCALER_PATH)
    with mlflow.start_run(run_name="base-train"):
        model = fit_base(train, scaler)
        m = metrics([s["actual_correction"] for s in val], predict(model, val, scaler))
        mlflow.log_params({"mode": "scratch", "n_train": len(train), "n_val": len(val), "seq_len": 20,
                           "group": "항로", "epochs_run": len(model.history.history["loss"])})
        mlflow.log_metrics(m)
        mlflow.tensorflow.log_model(model, name="model", input_example=build_sequences(train[:1], scaler)[0])
        return _register_if_gate_passed(mlflow.active_run().info.run_id, m)


def _archived_candidates(X_te, y_te, scaler, k: int = 3) -> list[dict]:
    """v1.5: 예전 버전(최근 k개, Archived)도 같은 홀드아웃에서 채점 → 되돌리기(rollback) 후보."""
    out = []
    try:
        vs = [v for v in MlflowClient().search_model_versions(f"name='{MODEL_NAME}'") if v.current_stage == "Archived"]
        for v in sorted(vs, key=lambda v: int(v.version))[-k:]:
            mdl = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/{v.version}")
            p = [scaler.inverse_target(float(x)) for x in mdl.predict(X_te, verbose=0).flatten()]
            out.append({"version": v.version, **metrics(y_te, p)})
    except Exception as e:
        print(f"[rollback] archived check skipped: {e}")
    return out


def fine_tune(recent: list[dict]) -> dict:
    """Production 가중치에서 이어서(warm start) 최근 배들로 짧게 학습.
    v1.5: 새 모델이 게이트를 못 넘거나 예전 버전이 더 나으면, 같은 홀드아웃에서 가장 나은 예전 버전으로 되돌린다.
    (드리프트가 일시적이었을 때 — 예: 시뮬레이션 뒤 실제 데이터로 복귀 — 학습 없이 바로 원래 모델로 돌아감)"""
    scaler = ArrivalScaler.load(SCALER_PATH)
    X, y = build_sequences(recent, scaler)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y)
    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    prod_preds = [scaler.inverse_target(float(p)) for p in model.predict(X_te, verbose=0).flatten()]
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")
    olds = _archived_candidates(X_te, y_te, scaler)
    best_old = min(olds, key=lambda o: o["mae"]) if olds else None
    with mlflow.start_run(run_name="fine-tune"):
        model.fit(X_tr, np.array([scaler.scale_target(v) for v in y_tr], dtype="float32"),
                  epochs=FINE_TUNE_EPOCHS, batch_size=32, verbose=0)
        preds = [scaler.inverse_target(float(p)) for p in model.predict(X_te, verbose=0).flatten()]
        m = {**metrics(y_te, preds), "prod_mae": metrics(y_te, prod_preds)["mae"]}
        if best_old:
            m["best_archived_mae"] = best_old["mae"]
        mlflow.log_params({"mode": "fine-tune", "epochs": FINE_TUNE_EPOCHS, "n_cases": len(recent),
                           "best_archived_version": best_old["version"] if best_old else None})
        mlflow.log_metrics(m)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_tr[:1])
        new_ok = _gate_fine_tune(m) and (best_old is None or m["mae"] <= best_old["mae"])
        if new_ok:
            return _register_if_gate_passed(mlflow.active_run().info.run_id, m, gate=_gate_fine_tune)
        if best_old and best_old["mae"] <= m["prod_mae"] and best_old["mae"] <= ABS_MAE_GATE:
            MlflowClient().transition_model_version_stage(name=MODEL_NAME, version=best_old["version"],
                                                          stage="Production", archive_existing_versions=True)
            print(f"[ROLLBACK] v{best_old['version']} mae={best_old['mae']:.3f}h (new {m['mae']:.3f}h, "
                  f"current prod {m['prod_mae']:.3f}h) -> back to Production")
            return {**m, "promoted": True, "rollback": True, "version": best_old["version"],
                    "mae": best_old["mae"], "rmse": best_old["rmse"]}
        return _register_if_gate_passed(mlflow.active_run().info.run_id, m, gate=_gate_fine_tune)


if __name__ == "__main__":
    train_and_register()
