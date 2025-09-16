# test_xgb_fallback.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, callback

X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=5,
    n_classes=2,
    random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42
)

print("Próba trenowania z callbackiem EarlyStopping...")
try:
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[callback.EarlyStopping(
            rounds=10,
            save_best=True,
            maximize=True
        )],
        verbose=True
    )
    print("✅ Działa na callbacks")
except TypeError as e:
    print("⚠️ Callbacks nie działają, próbuję early_stopping_rounds...")
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=True
    )
    print("✅ Działa na early_stopping_rounds")
