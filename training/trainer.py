"""
Model training and comparison module for IoT Intrusion Detection.
Trains Random Forest, Decision Tree, SVM, KNN, and XGBoost classifiers.
Measures wall-clock training and prediction times.
"""
import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from evaluation.evaluator import ModelEvaluator
from config import Config


class ModelTrainer:
    """Trainer orchestrator for multiple ML algorithms with standardized evaluation."""

    def __init__(self, X_train, X_test, y_train, y_test, feature_names=None, label_classes=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names or []
        self.label_classes = label_classes or []
        self.models = {}
        self.results = {}

    def _get_models(self):
        """Instantiate all 5 machine learning classifiers with tuned hyperparameters."""
        num_classes = len(np.unique(self.y_train))
        is_multiclass = num_classes > 2

        # XGBoost objective setup
        xgb_objective = 'multi:softprob' if is_multiclass else 'binary:logistic'

        return {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                random_state=Config.RANDOM_STATE
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=Config.RANDOM_STATE
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=Config.RANDOM_STATE,
                objective=xgb_objective,
                eval_metric='mlogloss' if is_multiclass else 'logloss',
                n_jobs=-1
            )
        }

    def train_model(self, model_name, model):
        """Train a single model and record execution time."""
        # Subsample for SVM if dataset is large to maintain interactive responsiveness
        if model_name == 'Support Vector Machine' and len(self.X_train) > 5000:
            indices = np.random.RandomState(Config.RANDOM_STATE).choice(len(self.X_train), 5000, replace=False)
            X_fit = self.X_train[indices]
            y_fit = self.y_train[indices]
        elif model_name == 'K-Nearest Neighbors' and len(self.X_train) > 10000:
            indices = np.random.RandomState(Config.RANDOM_STATE).choice(len(self.X_train), 10000, replace=False)
            X_fit = self.X_train[indices]
            y_fit = self.y_train[indices]
        else:
            X_fit = self.X_train
            y_fit = self.y_train

        start_time = time.time()
        model.fit(X_fit, y_fit)
        training_time = time.time() - start_time

        return model, round(training_time, 4)

    def evaluate_model(self, model_name, model, training_time):
        """Evaluate a trained model on unseen test data and calculate prediction time."""
        start_time = time.time()
        _ = model.predict(self.X_test)
        prediction_time = time.time() - start_time

        metrics = ModelEvaluator.evaluate(model, self.X_test, self.y_test, self.label_classes)
        metrics['training_time'] = training_time
        metrics['prediction_time'] = round(prediction_time, 4)
        metrics['model_name'] = model_name

        return metrics

    def train_all(self):
        """Train and evaluate all 5 algorithms sequentially, storing models and metrics."""
        candidate_models = self._get_models()

        for name, model_instance in candidate_models.items():
            fitted_model, tr_time = self.train_model(name, model_instance)
            self.models[name] = fitted_model
            metrics = self.evaluate_model(name, fitted_model, tr_time)
            self.results[name] = metrics

        return self.results

    def get_comparison(self):
        """Return models sorted by F1-Score descending."""
        comparison_list = list(self.results.values())
        comparison_list.sort(key=lambda x: (x.get('f1_score', 0), x.get('accuracy', 0)), reverse=True)
        return comparison_list

    def get_best_model(self):
        """Select top model based on composite F1-score and accuracy."""
        sorted_models = self.get_comparison()
        if not sorted_models:
            return None, None, {}
        best_metrics = sorted_models[0]
        best_name = best_metrics['model_name']
        best_model = self.models.get(best_name)
        return best_name, best_model, best_metrics

    def save_models(self, directory=Config.MODELS_FOLDER):
        """Serialize all trained models with joblib."""
        os.makedirs(directory, exist_ok=True)
        saved_paths = {}

        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}.joblib"
            filepath = os.path.join(directory, filename)
            joblib.dump(model, filepath)
            saved_paths[name] = filepath

        return saved_paths

    def save_model(self, model_name, filepath):
        """Save a specific model instance to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' has not been trained.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.models[model_name], filepath)
        return filepath
