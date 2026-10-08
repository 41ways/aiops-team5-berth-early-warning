"""도착 모델 시드 안정성: 조기 종료 기준 2가지 × 시드 3개 (테스트 MAE)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from tensorflow import keras
from data.features import ArrivalScaler, build_sequences, load_samples, split_by_period
from serving_app.lstm_model import build_model
sc = ArrivalScaler()
tr, va, te = split_by_period(load_samples(sys.argv[1]))
Xtr, ytr = build_sequences(tr, sc); Xva, yva = build_sequences(va, sc); Xte, yte = build_sequences(te, sc)
s = lambda y: np.array([sc.scale_target(v) for v in y], dtype="float32")
mae = lambda m, X, y: float(np.mean(np.abs(np.array(y) - np.array([sc.inverse_target(p) for p in m.predict(X, verbose=0).flatten()]))))
print("terminal plan test MAE", round(float(np.mean(np.abs(yte))), 3))
for es in ("val_split_0.15", "val_period"):
    r = []
    for seed in (1, 2, 3):
        keras.utils.set_random_seed(seed); m = build_model()
        kw = dict(validation_split=0.15) if es == "val_split_0.15" else dict(validation_data=(Xva, s(yva)))
        m.fit(Xtr, s(ytr), epochs=100, batch_size=32, verbose=0, callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)], **kw)
        r.append((round(mae(m, Xva, yva), 3), round(mae(m, Xte, yte), 3)))
    print(es, "(val, test) per seed:", r, flush=True)
