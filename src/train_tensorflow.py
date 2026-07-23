"""
TensorFlow/Keras churn model for the Chewy project.
Run:  python train_tensorflow.py  (expects ../data/chewy_features.parquet or the CSV)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)


def load_features():
    try:
        return pd.read_parquet("../data/chewy_features.parquet")
    except Exception:
        return pd.read_csv("../data/chewy_customers.csv").drop_duplicates("customer_id")


def prepare(df):
    cat_cols = [c for c in ["pet_type", "region", "plan", "primary_category"] if c in df]
    X = pd.get_dummies(df.drop(columns=["churned", "customer_id"]), columns=cat_cols, dtype=float)
    X = X.fillna(X.median(numeric_only=True))
    y = df["churned"].values.astype(np.float32)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X.values, y, test_size=0.15, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=SEED)
    sc = StandardScaler().fit(X_tr)
    return [sc.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te)], (y_tr, y_va, y_te)


def main():
    (X_tr, X_va, X_te), (y_tr, y_va, y_te) = prepare(load_features())
    pos_weight = float((len(y_tr) - y_tr.sum()) / y_tr.sum())

    model = keras.Sequential([
        layers.Input(shape=(X_tr.shape[1],)),
        layers.Dense(128, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=60, batch_size=512, verbose=0,
              class_weight={0: 1.0, 1: pos_weight},
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                       patience=8, restore_best_weights=True)])

    proba = model.predict(X_te, verbose=0).ravel()
    print(f"Test ROC-AUC: {roc_auc_score(y_te, proba):.4f} | PR-AUC: {average_precision_score(y_te, proba):.4f}")
    print(classification_report(y_te, (proba >= 0.5).astype(int), target_names=["Retained", "Churned"]))
    model.save("../models/chewy_churn_tensorflow.keras")
    print("Saved ../models/chewy_churn_tensorflow.keras")


if __name__ == "__main__":
    main()
