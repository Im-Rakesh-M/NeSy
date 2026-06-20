import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


class MachineForecaster:
    """
    XGBoost classifier with Conformal Prediction for
    machine failure prediction on AI4I dataset.

    Feature rationale:
    - thermal_stress      : process-air temp delta.
                            High delta = thermal runaway risk.
    - mechanical_load     : torque x rpm.
                            Physical strain proxy.
    - wear_criticality    : tool_wear / 240 (normalized).
                            Above 0.81 = danger zone per calibration.
    - failure_risk_score  : sum of all failure flags.
                            >1 = multiple systems failing.
    - jit_supply_pressure : rolling 5-step mean of risk score.
                            Captures sustained disruption pressure.
    - product_type_encoded: L=0, M=1, H=2.
                            H-type machines have higher failure rates.
    """

    FEATURE_COLS = [
        'thermal_stress',
        'mechanical_load',
        'wear_criticality',
        'failure_risk_score',
        'jit_supply_pressure',
        'product_type_encoded'
    ]

    MAX_TOOL_WEAR = 240

    def __init__(self, alpha=0.05):
        """
        alpha=0.05 gives 95% coverage guarantee.
        Stricter than delivery forecaster because machine
        failure has higher safety consequences.
        """
        self.alpha = alpha
        self.model = None
        self.cal_scores = None
        self.q_hat = None

    def preprocess(self, df):
        df = df.copy()

        df['thermal_stress'] = (
            df['Process temperature [K]'] -
            df['Air temperature [K]']
        ).round(3)

        df['mechanical_load'] = (
            df['Torque [Nm]'] * df['Rotational speed [rpm]']
        ).round(2)

        df['wear_criticality'] = (
            df['Tool wear [min]'] / self.MAX_TOOL_WEAR
        ).round(4)

        df['failure_risk_score'] = (
            df['TWF'] + df['HDF'] +
            df['PWF'] + df['OSF'] + df['RNF']
        )

        df['jit_supply_pressure'] = (
            df['failure_risk_score']
            .rolling(window=5, min_periods=1)
            .mean()
            .round(4)
        )

        type_map = {'L': 0, 'M': 1, 'H': 2}
        df['product_type_encoded'] = df['Type'].map(type_map).fillna(0)

        df = df.dropna(subset=self.FEATURE_COLS + ['Machine failure'])

        X = df[self.FEATURE_COLS].copy()
        y = df['Machine failure'].astype(int)

        return X, y

    def train(self, data_path="data/ai4i2020.csv"):
        print("[MACHINE FORECASTER] Loading AI4I dataset...")
        df = pd.read_csv(data_path)
        print(f"  -> Shape: {df.shape}")

        X, y = self.preprocess(df)
        print(f"  -> Features: {X.shape}")
        print(f"  -> Failure rate: {y.mean()*100:.1f}%")

        # Three-way split: 60% train, 20% calibration, 20% test
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.40,
            random_state=42, stratify=y
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50,
            random_state=42, stratify=y_tmp
        )

        print(f"  -> Train: {len(X_train)}, "
              f"Cal: {len(X_cal)}, "
              f"Test: {len(X_test)}")

        # Handle severe class imbalance (3.4% failure rate)
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"  -> Class weight: {pos_weight:.2f}")

        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
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

        print(f"\n[MACHINE FORECASTER] Test Performance:")
        print(f"  -> ROC-AUC: {auc:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=['Healthy', 'Failure']
        ))

        # Conformal calibration
        # Non-conformity score = 1 - P(true class | x)
        cal_probs = self.model.predict_proba(X_cal)
        cal_true_probs = cal_probs[
            np.arange(len(y_cal)), y_cal.values
        ]
        self.cal_scores = 1.0 - cal_true_probs

        # Compute q_hat once — stored, never recomputed per prediction
        n = len(self.cal_scores)
        q_level = min(
            np.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0
        )
        self.q_hat = float(np.quantile(self.cal_scores, q_level))

        print(f"\n[MACHINE FORECASTER] Conformal Calibration:")
        print(f"  -> Calibration samples : {n}")
        print(f"  -> Alpha (significance): {self.alpha}")
        print(f"  -> q_hat (threshold)   : {self.q_hat:.4f}")
        print(f"  -> Coverage guarantee  : {(1-self.alpha)*100:.0f}%")

    def predict(self, features: dict) -> dict:
        """
        Predicts machine failure probability with conformal
        prediction set.

        Args:
            features: dict with keys matching FEATURE_COLS

        Returns:
            dict with:
                failure_probability: float
                prediction_set: list
                confidence: HIGH/LOW/ABSTAIN
                risk_level: CRITICAL/HIGH/MEDIUM/LOW
        """
        if self.model is None or self.q_hat is None:
            raise RuntimeError(
                "Model not trained. Call train() or load()."
            )

        df = pd.DataFrame([features])

        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0

        df = df[self.FEATURE_COLS]

        probs = self.model.predict_proba(df)[0]
        prob_fail = float(probs[1])

        # Non-conformity scores per class
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

        # Risk level for agent decision making
        if prob_fail >= 0.80:
            risk_level = "CRITICAL"
        elif prob_fail >= 0.50:
            risk_level = "HIGH"
        elif prob_fail >= 0.25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "failure_probability": round(prob_fail, 4),
            "prediction_set": prediction_set,
            "confidence": confidence,
            "risk_level": risk_level,
            "q_hat": round(self.q_hat, 4),
            "will_fail": prob_fail > 0.5
        }

    def save(self, path="data/machine_forecaster.pkl"):
        joblib.dump({
            "model": self.model,
            "cal_scores": self.cal_scores,
            "q_hat": self.q_hat,
            "alpha": self.alpha
        }, path)
        print(f"[MACHINE FORECASTER] Saved to {path}")

    def load(self, path="data/machine_forecaster.pkl"):
        data = joblib.load(path)
        self.model = data["model"]
        self.cal_scores = data["cal_scores"]
        self.q_hat = data["q_hat"]
        self.alpha = data["alpha"]
        print(f"[MACHINE FORECASTER] Loaded from {path}")


if __name__ == "__main__":
    forecaster = MachineForecaster(alpha=0.05)
    forecaster.train()
    forecaster.save()

    # Test prediction
    test_input = {
        "thermal_stress": 11.5,
        "mechanical_load": 75000.0,
        "wear_criticality": 0.85,
        "failure_risk_score": 2,
        "jit_supply_pressure": 0.8,
        "product_type_encoded": 2
    }
    result = forecaster.predict(test_input)
    print(f"\n[TEST PREDICTION]")
    print(f"  Failure probability : {result['failure_probability']}")
    print(f"  Prediction set      : {result['prediction_set']}")
    print(f"  Confidence          : {result['confidence']}")
    print(f"  Risk level          : {result['risk_level']}")
    print(f"  Will fail           : {result['will_fail']}")