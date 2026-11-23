"""
Comprehensive automated unit and integration tests for the IoT Intrusion Detection System.
Tests database operations, preprocessing pipelines, ML model training,
detection engine, and Flask HTTP/API endpoints.
"""
import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

# Add root folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from database.db import (
    init_db, get_db, create_user, get_user_by_username,
    get_user_by_id, verify_password, save_model_info, set_best_model,
    get_all_models, get_best_model, save_detection_result, get_detection_history,
    save_alert, get_alerts, mark_alert_reviewed, get_dashboard_stats
)
from preprocessing.preprocessor import DataPreprocessor
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from detection.detector import IntrusionDetector
from app import app as flask_app


@pytest.fixture(scope="session")
def synthetic_data_path(tmp_path_factory):
    """Generate temporary synthetic CSV dataset for testing."""
    tmp_dir = tmp_path_factory.mktemp("data")
    file_path = os.path.join(tmp_dir, "test_network.csv")

    np.random.seed(42)
    n = 120
    data = {
        'flow_duration': np.random.uniform(0.1, 10.0, n),
        'Header_Length': np.random.randint(20, 60, n),
        'Protocol_Type': np.random.choice([6, 17], n),
        'Rate': np.random.uniform(10.0, 5000.0, n),
        'Srate': np.random.uniform(5.0, 2500.0, n),
        'Drate': np.random.uniform(5.0, 2500.0, n),
        'syn_flag_number': np.random.choice([0, 1], n),
        'ack_flag_number': np.random.choice([0, 1], n),
        'Tot_sum': np.random.uniform(100.0, 10000.0, n),
        'Tot_size': np.random.uniform(40.0, 1500.0, n),
        'IAT': np.random.uniform(0.0001, 0.1, n),
        'label': np.random.choice(['BenignTraffic', 'DDoS-TCP_Flood', 'Mirai-greeth_flood'], n)
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def client():
    """Configure Flask test client with test configuration."""
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False
    with flask_app.test_client() as client:
        with flask_app.app_context():
            init_db()
        yield client


# ==========================================
# 1. DATABASE TESTS
# ==========================================

class TestDatabase:
    def test_database_initialization(self):
        init_db()
        admin = get_user_by_username('admin')
        assert admin is not None
        assert admin['username'] == 'admin'

    def test_create_and_verify_user(self):
        import time
        ts = int(time.time() * 1000)
        uname = f"analyst_{ts}"
        email = f"analyst_{ts}@cyber.org"
        uid = create_user(uname, email, "Pass1234!")
        assert uid is not None
        user = get_user_by_username(uname)
        assert user['email'] == email
        assert verify_password(user['password_hash'], "Pass1234!") is True
        assert verify_password(user['password_hash'], "WrongPass") is False

    def test_save_and_retrieve_model_metrics(self):
        metrics = {
            'accuracy': 0.985,
            'precision': 0.982,
            'recall': 0.985,
            'f1_score': 0.983,
            'roc_auc': 0.991,
            'training_time': 1.25,
            'prediction_time': 0.04
        }
        mid = save_model_info("Random Forest", metrics, "/fake/path/rf.joblib", ["f1", "f2"], ["Benign", "DDoS"])
        assert mid > 0
        set_best_model(mid)
        best = get_best_model()
        assert best is not None
        assert best['model_name'] == "Random Forest"

    def test_save_detection_and_alerts(self):
        det_id = save_detection_result(
            user_id=1,
            filename="batch_stream.csv",
            total=100,
            normal=80,
            malicious=20,
            rate=20.0,
            model_name="Random Forest"
        )
        assert det_id > 0

        alert_id = save_alert(det_id, "DDoS-SYN_Flood", 94.5, "192.168.1.15", "high")
        assert alert_id > 0

        alerts = get_alerts(limit=10)
        assert len(alerts) >= 1

        mark_alert_reviewed(alert_id)
        unreviewed = get_alerts(unreviewed_only=True)
        assert all(a['id'] != alert_id for a in unreviewed)

    def test_dashboard_stats(self):
        stats = get_dashboard_stats()
        assert 'total_records' in stats
        assert 'total_normal' in stats
        assert 'total_malicious' in stats
        assert 'detection_rate' in stats


# ==========================================
# 2. PREPROCESSING PIPELINE TESTS
# ==========================================

class TestPreprocessing:
    def test_pipeline_execution(self, synthetic_data_path, tmp_path):
        prep = DataPreprocessor()
        df = prep.load_data(synthetic_data_path)
        assert df is not None
        assert prep.label_column == 'label'

        summary = prep.run_full_pipeline(synthetic_data_path, test_size=0.2, max_features=8)
        assert summary['cleaned_rows'] > 0
        assert summary['features_count'] <= 8
        assert prep.X_train is not None
        assert prep.X_test is not None

        # Test pipeline serialization and reloading
        pipe_path = os.path.join(tmp_path, "pipeline.joblib")
        prep.save_pipeline(pipe_path)
        assert os.path.exists(pipe_path)

        loaded_pipe = DataPreprocessor.load_pipeline(pipe_path)
        assert 'scaler' in loaded_pipe
        assert 'selected_features' in loaded_pipe

        # Test inference transformation on unseen raw DataFrame
        test_df = pd.DataFrame([{
            'flow_duration': 1.5,
            'Header_Length': 32,
            'Protocol_Type': 6,
            'Rate': 150.0,
            'Srate': 75.0,
            'Drate': 75.0,
            'syn_flag_number': 0,
            'ack_flag_number': 1,
            'Tot_sum': 500.0,
            'Tot_size': 300.0,
            'IAT': 0.01
        }])
        X_trans = prep.transform_new_data(test_df)
        assert X_trans.shape[1] == len(prep.selected_features)


# ==========================================
# 3. TRAINING & EVALUATION TESTS
# ==========================================

class TestTrainingAndEvaluation:
    def test_train_all_algorithms(self, synthetic_data_path, tmp_path):
        prep = DataPreprocessor()
        prep.run_full_pipeline(synthetic_data_path, test_size=0.25, max_features=6)

        trainer = ModelTrainer(
            X_train=prep.X_train,
            X_test=prep.X_test,
            y_train=prep.y_train,
            y_test=prep.y_test,
            feature_names=prep.feature_names,
            label_classes=prep.label_classes
        )

        results = trainer.train_all()
        assert len(results) == 5
        assert 'Random Forest' in results
        assert 'Decision Tree' in results
        assert 'Support Vector Machine' in results
        assert 'K-Nearest Neighbors' in results
        assert 'XGBoost' in results

        best_name, best_model, best_metrics = trainer.get_best_model()
        assert best_name is not None
        assert best_model is not None
        assert 'accuracy' in best_metrics

        # Test model persistence
        saved_paths = trainer.save_models(str(tmp_path))
        assert os.path.exists(saved_paths['Random Forest'])


# ==========================================
# 4. INTRUSION DETECTION ENGINE TESTS
# ==========================================

class TestIntrusionDetection:
    def test_detector_inference(self, synthetic_data_path, tmp_path):
        prep = DataPreprocessor()
        prep.run_full_pipeline(synthetic_data_path, test_size=0.25, max_features=6)

        trainer = ModelTrainer(
            X_train=prep.X_train,
            X_test=prep.X_test,
            y_train=prep.y_train,
            y_test=prep.y_test,
            feature_names=prep.feature_names,
            label_classes=prep.label_classes
        )
        trainer.train_all()
        best_name, best_model, _ = trainer.get_best_model()

        model_path = os.path.join(tmp_path, "best_model.joblib")
        pipe_path = os.path.join(tmp_path, "pipeline.joblib")
        trainer.save_model(best_name, model_path)
        prep.save_pipeline(pipe_path)

        detector = IntrusionDetector(model_path=model_path, pipeline_path=pipe_path)
        test_df = pd.read_csv(synthetic_data_path).head(15)

        res = detector.predict_with_confidence(test_df)
        assert res['total_records'] == 15
        assert len(res['records']) == 15
        assert 'confidence' in res['records'][0]

        alerts = detector.generate_alerts(res['records'], threshold=50.0)
        assert isinstance(alerts, list)

        report_file = os.path.join(tmp_path, "report.csv")
        detector.generate_report(test_df, res['records'], report_file)
        assert os.path.exists(report_file)


# ==========================================
# 5. FLASK WEB ROUTES & APIS
# ==========================================

class TestWebRoutes:
    def test_unauthenticated_redirect(self, client):
        res = client.get('/dashboard', follow_redirects=False)
        assert res.status_code == 302
        assert '/login' in res.location

    def test_login_and_authenticated_access(self, client):
        # Login with default admin
        res = client.post('/login', data={'username': 'admin', 'password': 'admin123'}, follow_redirects=True)
        assert res.status_code == 200
        assert b"IoT Intrusion Command Center" in res.data

        # Access dashboard
        dash = client.get('/dashboard')
        assert dash.status_code == 200

        # Access dataset manager
        dataset_view = client.get('/dataset')
        assert dataset_view.status_code == 200

        # Access model comparison
        comp_view = client.get('/model-comparison')
        assert comp_view.status_code == 200

        # API alerts data
        alerts_api = client.get('/api/alerts/data')
        assert alerts_api.status_code == 200
        json_data = alerts_api.get_json()
        assert json_data['success'] is True
