"""
Model evaluation module for IoT Intrusion Detection System.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:
    """Evaluator for calculating and standardizing multi-class and binary metrics."""

    @staticmethod
    def evaluate(model, X_test, y_test, label_classes=None):
        """Compute complete suite of classification performance metrics."""
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # ROC-AUC calculation
        roc_auc = ModelEvaluator.get_roc_auc(model, X_test, y_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Classification report dict
        try:
            target_names = [str(c) for c in label_classes] if label_classes and len(label_classes) == len(np.unique(y_test)) else None
            cls_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        except Exception:
            cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        return {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4),
            'confusion_matrix': cm,
            'classification_report': cls_report
        }

    @staticmethod
    def get_confusion_matrix(model, X_test, y_test):
        """Return raw confusion matrix."""
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    @staticmethod
    def get_classification_report(model, X_test, y_test, label_classes=None):
        """Return detailed classification report dictionary."""
        y_pred = model.predict(X_test)
        target_names = [str(c) for c in label_classes] if label_classes else None
        return classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    @staticmethod
    def get_roc_auc(model, X_test, y_test):
        """Calculate ROC-AUC score handling both binary and multiclass (OVR)."""
        try:
            if not hasattr(model, 'predict_proba'):
                return 0.0
            y_proba = model.predict_proba(X_test)
            unique_classes = np.unique(y_test)

            if len(unique_classes) == 2:
                # Binary classification
                if y_proba.shape[1] == 2:
                    return float(roc_auc_score(y_test, y_proba[:, 1]))
                return float(roc_auc_score(y_test, y_proba))
            elif len(unique_classes) > 2:
                # Multiclass classification using One-vs-Rest (OVR)
                return float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted'))
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def compare_models(results_dict):
        """Format comparison dictionary sorted by performance."""
        items = list(results_dict.values()) if isinstance(results_dict, dict) else results_dict
        return sorted(items, key=lambda x: x.get('f1_score', 0), reverse=True)

    @staticmethod
    def generate_report(results_dict, filepath):
        """Export model comparison table to CSV."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        rows = []
        for name, m in results_dict.items():
            rows.append({
                'Model': name,
                'Accuracy': m.get('accuracy', 0),
                'Precision': m.get('precision', 0),
                'Recall': m.get('recall', 0),
                'F1 Score': m.get('f1_score', 0),
                'ROC AUC': m.get('roc_auc', 0),
                'Training Time (s)': m.get('training_time', 0),
                'Prediction Time (s)': m.get('prediction_time', 0)
            })
        df = pd.DataFrame(rows)
        df.sort_values(by='F1 Score', ascending=False, inplace=True)
        df.to_csv(filepath, index=False)
        return filepath
