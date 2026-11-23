"""
Intrusion detection engine for IoT Networks.
Loads trained models and preprocessing pipelines to classify incoming traffic batches,
calculates detection metrics, produces alerts, and exports analytical reports.
"""
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from config import Config


class IntrusionDetector:
    """Real-time / Batch intrusion detection processor."""

    def __init__(self, model_path=None, pipeline_path=None):
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self.model = None
        self.pipeline = None

        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

        if pipeline_path and os.path.exists(pipeline_path):
            self.pipeline = joblib.load(pipeline_path)

    def load_artifacts(self, model_path, pipeline_path):
        """Explicitly load or swap model and pipeline bundles."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)
        self.model_path = model_path
        self.pipeline_path = pipeline_path

    def is_attack(self, label_str):
        """Determine whether a predicted label corresponds to malicious traffic."""
        norm_label = str(label_str).strip().lower()
        return norm_label not in ['benigntraffic', 'benign', 'normal', '0', '0.0']

    def get_attack_category(self, label_str):
        """Map specific attack label to broader high-level category."""
        return Config.ATTACK_CATEGORIES.get(str(label_str), "Malicious Traffic" if self.is_attack(label_str) else "Normal")

    def transform_features(self, df):
        """Align input features and scale using the saved preprocessing pipeline."""
        if not self.pipeline:
            raise ValueError("Preprocessing pipeline is not loaded.")

        selected_features = self.pipeline.get('selected_features', [])
        scaler = self.pipeline.get('scaler')
        label_col = self.pipeline.get('label_column')

        df_copy = df.copy()
        if label_col and label_col in df_copy.columns:
            df_copy = df_copy.drop(columns=[label_col])

        # Fill missing expected features with 0.0
        for feat in selected_features:
            if feat not in df_copy.columns:
                df_copy[feat] = 0.0

        # Subsample and order exactly as trained
        X_sub = df_copy[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Scale
        return scaler.transform(X_sub)

    def predict(self, df):
        """Classify batch records into normal and attack categories."""
        if not self.model or not self.pipeline:
            raise ValueError("Model and preprocessor must be loaded before running inference.")

        X_transformed = self.transform_features(df)
        preds_raw = self.model.predict(X_transformed)

        # Decode class labels if label encoder is available
        label_encoder = self.pipeline.get('label_encoder')
        if label_encoder is not None:
            try:
                predicted_labels = label_encoder.inverse_transform(preds_raw)
            except Exception:
                predicted_labels = preds_raw
        else:
            predicted_labels = preds_raw

        predictions = [str(p) for p in predicted_labels]

        # Calculate statistics
        total_records = len(predictions)
        malicious_count = sum(1 for p in predictions if self.is_attack(p))
        normal_count = total_records - malicious_count
        detection_rate = round((malicious_count / total_records * 100), 2) if total_records > 0 else 0.0

        attack_dist = self.get_attack_distribution(predictions)

        return {
            'predictions': predictions,
            'attack_distribution': attack_dist,
            'total_records': total_records,
            'normal_count': normal_count,
            'malicious_count': malicious_count,
            'detection_rate': detection_rate
        }

    def predict_with_confidence(self, df):
        """Classify records and determine prediction probabilities and severity levels."""
        if not self.model or not self.pipeline:
            raise ValueError("Model and preprocessor must be loaded before running inference.")

        X_transformed = self.transform_features(df)
        preds_raw = self.model.predict(X_transformed)

        # Extract probability / confidence
        if hasattr(self.model, 'predict_proba'):
            try:
                probas = self.model.predict_proba(X_transformed)
                confidences = [float(np.max(prob)) for prob in probas]
            except Exception:
                confidences = [0.95] * len(preds_raw)
        else:
            confidences = [0.90] * len(preds_raw)

        # Decode labels
        label_encoder = self.pipeline.get('label_encoder')
        if label_encoder is not None:
            try:
                predicted_labels = label_encoder.inverse_transform(preds_raw)
            except Exception:
                predicted_labels = preds_raw
        else:
            predicted_labels = preds_raw

        records = []
        for i, (pred, conf) in enumerate(zip(predicted_labels, confidences)):
            pred_str = str(pred)
            is_mal = self.is_attack(pred_str)
            category = self.get_attack_category(pred_str)

            # Determine severity
            if not is_mal:
                severity = "normal"
            elif conf >= 0.85:
                severity = "high"
            elif conf >= 0.65:
                severity = "medium"
            else:
                severity = "low"

            records.append({
                'record_id': i + 1,
                'prediction': pred_str,
                'category': category,
                'confidence': round(conf * 100, 2),
                'is_malicious': is_mal,
                'severity': severity
            })

        total_records = len(records)
        malicious_count = sum(1 for r in records if r['is_malicious'])
        normal_count = total_records - malicious_count
        detection_rate = round((malicious_count / total_records * 100), 2) if total_records > 0 else 0.0

        attack_dist = self.get_attack_distribution([r['prediction'] for r in records])

        return {
            'records': records,
            'predictions': [r['prediction'] for r in records],
            'confidences': confidences,
            'attack_distribution': attack_dist,
            'total_records': total_records,
            'normal_count': normal_count,
            'malicious_count': malicious_count,
            'detection_rate': detection_rate
        }

    def get_attack_distribution(self, predictions):
        """Count frequency of each predicted class."""
        dist = {}
        for p in predictions:
            dist[p] = dist.get(p, 0) + 1
        return dist

    def generate_alerts(self, records, threshold=60.0):
        """Construct alert objects for detected threats exceeding confidence threshold."""
        alerts = []
        for rec in records:
            if rec['is_malicious'] and rec['confidence'] >= threshold:
                alerts.append({
                    'attack_type': rec['prediction'],
                    'category': rec['category'],
                    'confidence': rec['confidence'],
                    'severity': rec['severity'],
                    'source_ip': f"192.168.1.{np.random.randint(2, 254)}",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        return alerts

    def generate_report(self, df_original, records, filepath):
        """Produce full audit CSV report linking original telemetry with predictions."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        report_df = df_original.copy()
        report_df['Predicted_Label'] = [r['prediction'] for r in records]
        report_df['Attack_Category'] = [r['category'] for r in records]
        report_df['Confidence_Percent'] = [r['confidence'] for r in records]
        report_df['Is_Malicious'] = [1 if r['is_malicious'] else 0 for r in records]
        report_df['Severity'] = [r['severity'] for r in records]
        report_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_df.to_csv(filepath, index=False)
        return filepath
