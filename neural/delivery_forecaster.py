import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score


class DeliveryForecaster:
    """
    XGBoost classifier with Conformal Prediction intervals
    for JIT delivery delay prediction.

    Conformal Prediction Rationale:
    Standard ML models output point probabilities with no
    statistical guarantees. Conformal prediction wraps the
    model and produces prediction SETS with guaranteed
    coverage — if alpha=0.10, the true label is in the
    prediction set at least 90% of the time, regardless
    of data distribution. This is the uncertainty
    quantification pillar of our trustworthy AI system.

    Split Conformal Method:
    We use three splits — train, calibration, test.
    Calibration must be separate from both train and test.
    Using test data for calibration invalidates the
    coverage guarantee mathematically.
    """

    FEATURE_COLS = [
        'urgency_score',
        'Order Item Quantity',
        'Order Region',
        'Shipping Mode',
        'Category Id',
        'delay_days'
    ]

    def __init__(self, alpha=0.10):
        """
        alpha: significance level.
        alpha=0.10 gives 90% coverage guarantee.
        """
        self.alpha = alpha
        self.model = None
        self.label_encoders = {}
        self.cal_scores = None
        self.q_hat = None

    def preprocess(self, df):
        """
        Engineers features and encodes categoricals.
        Uses DataCo's real Late_delivery_risk column as target —
        not a derived recalculation.
        """
        df = df.copy()

        # Feature engineering
        df['delay_days'] = (
            df['Days for shipping (real)'] -
            df['Days for shipment (scheduled)']
        )
        df['urgency_score'] = df['Shipping Mode'].map({
            'Same Day': 3,
            'First Class': 2,
            'Second Class': 1,
            'Standard Class': 0
        }).fillna(0).astype(int)

        # Drop nulls in required columns
        df = df.dropna(subset=self.FEATURE_COLS + ['Late_delivery_risk'])

        X = df[self.FEATURE_COLS].copy()
        y = df['Late_delivery_risk'].astype(int)

        # Encode categoricals
        for col in ['Order Region', 'Shipping Mode']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        X['Category Id'] = X['Category Id'].fillna(0).astype(int)

        return X, y

    def train(self, data_path="data/DataCoSupplyChainDataset.csv"):
        print("[DELIVERY FORECASTER] Loading DataCo dataset...")
        df = pd.read_csv(data_path, encoding='latin1')
        print(f"  -> Shape: {df.shape}")

        X, y = self.preprocess(df)
        print(f"  -> Features: {X.shape}, Target balance: "
              f"{y.mean()*100:.1f}% late")

        # Three-way split: 60% train, 20% calibration, 20% test
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.40, random_state=42, stratify=y
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
        )

        print(f"  -> Train: {len(X_train)}, "
              f"Cal: {len(X_cal)}, "
              f"Test: {len(X_test)}")

        # Handle class imbalance
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"  -> Class weight: {pos_weight:.2f}")

        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n[DELIVERY FORECASTER] Test Performance:")
        print(f"  -> ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred,
              target_names=['On Time', 'Late']))

        # Conformal calibration
        # Non-conformity score = 1 - P(true class | x)
        cal_probs = self.model.predict_proba(X_cal)
        cal_true_probs = cal_probs[
            np.arange(len(y_cal)), y_cal.values
        ]
        self.cal_scores = 1.0 - cal_true_probs

        # Compute quantile threshold
        n = len(self.cal_scores)
        q_level = min(
            np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0
        )
        self.q_hat = float(np.quantile(self.cal_scores, q_level))

        print(f"\n[DELIVERY FORECASTER] Conformal Calibration:")
        print(f"  -> Calibration samples : {n}")
        print(f"  -> Alpha (significance): {self.alpha}")
        print(f"  -> q_hat (threshold)   : {self.q_hat:.4f}")
        print(f"  -> Coverage guarantee  : {(1-self.alpha)*100:.0f}%")

    def predict(self, features: dict) -> dict:
        """
        Predicts delay probability with conformal prediction set.

        Args:
            features: dict with keys matching FEATURE_COLS
                     (Order Region and Shipping Mode as strings)

        Returns:
            dict with:
                late_probability: float
                prediction_set: list ([] = abstain, [0] = on time,
                                [1] = late, [0,1] = uncertain)
                confidence: str (HIGH/MEDIUM/LOW)
                q_hat: float
        """
        if self.model is None or self.q_hat is None:
            raise RuntimeError(
                "Model not trained. Call train() first or load()."
            )

        df = pd.DataFrame([features])

        # Apply label encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])
                else:
                    df[col] = 0  # Unknown category defaults to 0

        # Fill missing features with 0
        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0

        df = df[self.FEATURE_COLS]

        probs = self.model.predict_proba(df)[0]
        prob_late = float(probs[1])

        # Build conformal prediction set
        scores = 1.0 - probs
        prediction_set = [
            i for i, s in enumerate(scores) if s <= self.q_hat
        ]

        # Confidence interpretation
        if len(prediction_set) == 1:
            confidence = "HIGH"
        elif len(prediction_set) == 2:
            confidence = "LOW"
        else:
            confidence = "ABSTAIN"

        return {
            "late_probability": round(prob_late, 4),
            "prediction_set": prediction_set,
            "confidence": confidence,
            "q_hat": round(self.q_hat, 4),
            "will_be_late": prob_late > 0.5
        }

    def save(self, path="data/delivery_forecaster.pkl"):
        joblib.dump({
            "model": self.model,
            "label_encoders": self.label_encoders,
            "cal_scores": self.cal_scores,
            "q_hat": self.q_hat,
            "alpha": self.alpha
        }, path)
        print(f"[DELIVERY FORECASTER] Saved to {path}")

    def load(self, path="data/delivery_forecaster.pkl"):
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoders = data["label_encoders"]
        self.cal_scores = data["cal_scores"]
        self.q_hat = data["q_hat"]
        self.alpha = data["alpha"]
        print(f"[DELIVERY FORECASTER] Loaded from {path}")


if __name__ == "__main__":
    forecaster = DeliveryForecaster(alpha=0.10)
    forecaster.train()
    forecaster.save()

    # Test prediction
    test_input = {
        "urgency_score": 1,
        "Order Item Quantity": 10,
        "Order Region": "Western Europe",
        "Shipping Mode": "Standard Class",
        "Category Id": 24,
        "delay_days": 1.0
    }
    result = forecaster.predict(test_input)
    print(f"\n[TEST PREDICTION]")
    print(f"  Late probability : {result['late_probability']}")
    print(f"  Prediction set   : {result['prediction_set']}")
    print(f"  Confidence       : {result['confidence']}")
    print(f"  Will be late     : {result['will_be_late']}")