"""
Flask Web Application for Intelligent IoT Intrusion Detection System.
Provides end-to-end user interfaces, RESTful APIs, data management, model training,
real-time batch detection, cybersecurity dashboard, and alert management.
"""
import os
import json
import pandas as pd
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, send_file
)
from werkzeug.utils import secure_filename

from config import Config
from database.db import (
    init_db, get_user_by_username, get_user_by_id, create_user,
    verify_password, save_dataset_info, get_latest_dataset_info,
    save_preprocessing_info, get_latest_preprocessing_info,
    save_model_info, set_best_model, get_all_models, get_best_model,
    save_detection_result, get_detection_history, save_alert,
    get_alerts, mark_alert_reviewed, get_dashboard_stats
)
from preprocessing.preprocessor import DataPreprocessor
from training.trainer import ModelTrainer
from detection.detector import IntrusionDetector

app = Flask(__name__)
app.config.from_object(Config)

# Ensure essential storage folders exist
for folder in [Config.UPLOAD_FOLDER, Config.MODELS_FOLDER, Config.REPORTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize database schema
init_db()


def login_required(f):
    """Decorator to enforce session authentication on protected endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access the cybersecurity dashboard.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ==========================================
# AUTHENTICATION ROUTES
# ==========================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle administrative authentication."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = get_user_by_username(username)
        if user and verify_password(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user.get('role', 'admin')
            flash(f'Welcome back, {user["username"]}! Session initialized.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password credentials.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle administrator account registration."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not username or not email or not password:
            flash('All registration fields are required.', 'danger')
        elif password != confirm_password:
            flash('Password confirmation does not match.', 'danger')
        elif len(password) < 6:
            flash('Password must contain at least 6 characters.', 'danger')
        elif get_user_by_username(username):
            flash('Username is already registered. Choose another username.', 'warning')
        else:
            try:
                user_id = create_user(username, email, password)
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash(f'Registration error: {str(e)}', 'danger')

    return render_template('register.html')


@app.route('/logout')
def logout():
    """Terminate current user session."""
    session.clear()
    flash('You have been securely logged out.', 'info')
    return redirect(url_for('login'))


# ==========================================
# CORE DASHBOARD
# ==========================================

@app.route('/')
def index():
    """Root redirect to dashboard if authenticated or login otherwise."""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Cybersecurity command center dashboard."""
    stats = get_dashboard_stats()
    best_model = get_best_model()
    recent_alerts = stats.get('recent_alerts', [])
    return render_template('dashboard.html', stats=stats, best_model=best_model, recent_alerts=recent_alerts)


# ==========================================
# DATASET MANAGEMENT
# ==========================================

@app.route('/dataset')
@login_required
def dataset():
    """Dataset management and exploration view."""
    dataset_info = get_latest_dataset_info()
    return render_template('dataset.html', dataset_info=dataset_info)


@app.route('/dataset/upload', methods=['POST'])
@login_required
def dataset_upload():
    """Handle IoT telemetry CSV dataset upload."""
    if 'file' not in request.files:
        flash('No file part provided in the request.', 'danger')
        return redirect(url_for('dataset'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('dataset'))

    if not file.filename.lower().endswith('.csv'):
        flash('Only CSV telemetry files are permitted.', 'danger')
        return redirect(url_for('dataset'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        num_rows, num_cols = df.shape
        columns = list(df.columns)

        save_dataset_info(filename, num_rows, num_cols, file_size_mb, columns)
        flash(f'Dataset "{filename}" uploaded successfully ({num_rows} records, {num_cols} features).', 'success')
    except Exception as e:
        flash(f'Failed to process dataset file: {str(e)}', 'danger')

    return redirect(url_for('dataset'))


@app.route('/api/dataset/preview')
@login_required
def api_dataset_preview():
    """JSON API for viewing sample records of the current dataset."""
    info = get_latest_dataset_info()
    if not info:
        return jsonify({'success': False, 'message': 'No dataset uploaded yet.'})

    filepath = os.path.join(Config.UPLOAD_FOLDER, info['filename'])
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'Dataset file is missing from storage.'})

    try:
        df = pd.read_csv(filepath, nrows=20)
        df = df.fillna('N/A')
        return jsonify({
            'success': True,
            'columns': list(df.columns),
            'records': df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/dataset/info')
@login_required
def api_dataset_info():
    """JSON API for comprehensive dataset metadata."""
    info = get_latest_dataset_info()
    if not info:
        return jsonify({'success': False, 'message': 'No dataset uploaded.'})

    filepath = os.path.join(Config.UPLOAD_FOLDER, info['filename'])
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'Dataset file missing.'})

    try:
        preprocessor = DataPreprocessor()
        preprocessor.load_data(filepath)
        meta = preprocessor.get_info()
        return jsonify({'success': True, 'data': meta})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


# ==========================================
# PREPROCESSING MODULE
# ==========================================

@app.route('/preprocessing')
@login_required
def preprocessing():
    """Data preprocessing pipeline visualization page."""
    dataset_info = get_latest_dataset_info()
    preproc_info = get_latest_preprocessing_info()
    return render_template('preprocessing.html', dataset_info=dataset_info, preprocessing_info=preproc_info)


@app.route('/api/preprocessing/run', methods=['POST'])
@login_required
def api_preprocessing_run():
    """Execute end-to-end cleaning, encoding, selection, scaling, and splitting."""
    dataset_info = get_latest_dataset_info()
    if not dataset_info:
        return jsonify({'success': False, 'message': 'No dataset available to preprocess. Upload a CSV first.'}), 400

    filepath = os.path.join(Config.UPLOAD_FOLDER, dataset_info['filename'])
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'Dataset file missing from disk.'}), 404

    try:
        preprocessor = DataPreprocessor()
        summary = preprocessor.run_full_pipeline(
            filepath=filepath,
            test_size=Config.TEST_SIZE,
            max_features=Config.MAX_FEATURES
        )

        # Save fitted pipeline bundle
        pipeline_path = os.path.join(Config.MODELS_FOLDER, 'preprocessor_pipeline.joblib')
        preprocessor.save_pipeline(pipeline_path)

        # Save preprocessed splits for fast training reuse
        splits_path = os.path.join(Config.MODELS_FOLDER, 'preprocessed_splits.joblib')
        import joblib
        joblib.dump({
            'X_train': preprocessor.X_train,
            'X_test': preprocessor.X_test,
            'y_train': preprocessor.y_train,
            'y_test': preprocessor.y_test,
            'feature_names': preprocessor.feature_names,
            'label_classes': preprocessor.label_classes
        }, splits_path)

        # Save to database
        save_preprocessing_info(
            dataset_id=dataset_info['id'],
            missing_values=summary['missing_imputed'],
            duplicates_removed=summary['duplicates_removed'],
            features_selected=summary['selected_features'],
            scaling_method=summary['scaling_method'],
            label_column=summary['label_column'],
            num_classes=summary['num_classes'],
            class_names=summary['classes']
        )

        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Preprocessing error: {str(e)}'}), 500


# ==========================================
# ML MODEL TRAINING
# ==========================================

@app.route('/training')
@login_required
def training():
    """Model training control page."""
    models = get_all_models()
    best_model = get_best_model()
    return render_template('training.html', models=models, best_model=best_model)


@app.route('/api/training/run', methods=['POST'])
@login_required
def api_training_run():
    """Train all 5 algorithms, evaluate on validation data, and persist to disk and DB."""
    splits_path = os.path.join(Config.MODELS_FOLDER, 'preprocessed_splits.joblib')
    if not os.path.exists(splits_path):
        return jsonify({'success': False, 'message': 'Preprocessing must be executed before model training.'}), 400

    try:
        import joblib
        splits = joblib.load(splits_path)

        trainer = ModelTrainer(
            X_train=splits['X_train'],
            X_test=splits['X_test'],
            y_train=splits['y_train'],
            y_test=splits['y_test'],
            feature_names=splits.get('feature_names', []),
            label_classes=splits.get('label_classes', [])
        )

        results = trainer.train_all()
        saved_paths = trainer.save_models(Config.MODELS_FOLDER)

        best_name, best_inst, best_metrics = trainer.get_best_model()
        best_model_id = None

        # Insert each model record into DB
        for name, metrics in results.items():
            model_path = saved_paths.get(name, '')
            model_id = save_model_info(
                model_name=name,
                metrics_dict=metrics,
                model_path=model_path,
                feature_names=splits.get('feature_names', []),
                label_classes=splits.get('label_classes', [])
            )
            if name == best_name:
                best_model_id = model_id

        if best_model_id:
            set_best_model(best_model_id)

        return jsonify({
            'success': True,
            'best_model': best_name,
            'comparison': trainer.get_comparison()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Training error: {str(e)}'}), 500


# ==========================================
# MODEL COMPARISON
# ==========================================

@app.route('/model-comparison')
@login_required
def model_comparison():
    """Visual comparison of all trained ML models."""
    models = get_all_models()
    best_model = get_best_model()
    return render_template('model_comparison.html', models=models, best_model=best_model)


@app.route('/api/models/comparison')
@login_required
def api_models_comparison():
    """JSON API providing comparison metrics across all algorithms."""
    models = get_all_models()
    best_model = get_best_model()
    return jsonify({
        'success': True,
        'models': models,
        'best_model': best_model
    })


# ==========================================
# INTRUSION DETECTION
# ==========================================

@app.route('/detection')
@login_required
def detection():
    """Real-time batch intrusion detection view."""
    history = get_detection_history(limit=10)
    best_model = get_best_model()
    return render_template('detection.html', detection_history=history, best_model=best_model)


@app.route('/api/detection/predict', methods=['POST'])
@login_required
def api_detection_predict():
    """Classify uploaded IoT network traffic batch using optimal trained model."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No CSV file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file chosen.'}), 400

    best_model = get_best_model()
    if not best_model:
        return jsonify({'success': False, 'message': 'No trained models available. Please train models first.'}), 400

    pipeline_path = os.path.join(Config.MODELS_FOLDER, 'preprocessor_pipeline.joblib')
    if not os.path.exists(pipeline_path):
        return jsonify({'success': False, 'message': 'Preprocessing pipeline not found. Run preprocessing first.'}), 400

    try:
        filename = secure_filename(file.filename)
        upload_dest = os.path.join(Config.UPLOAD_FOLDER, f"detect_{filename}")
        file.save(upload_dest)

        df = pd.read_csv(upload_dest)

        # Initialize detector with active best model
        detector = IntrusionDetector(
            model_path=best_model['model_path'],
            pipeline_path=pipeline_path
        )

        results = detector.predict_with_confidence(df)

        # Generate audit report
        report_filename = f"report_{os.path.splitext(filename)[0]}_{int(pd.Timestamp.now().timestamp())}.csv"
        report_path = os.path.join(Config.REPORTS_FOLDER, report_filename)
        detector.generate_report(df, results['records'], report_path)

        # Save detection batch record
        detection_id = save_detection_result(
            user_id=session.get('user_id'),
            filename=filename,
            total=results['total_records'],
            normal=results['normal_count'],
            malicious=results['malicious_count'],
            rate=results['detection_rate'],
            model_used=best_model['model_name'],
            report_path=report_path
        )

        # Generate and save intrusion alerts
        alerts_list = detector.generate_alerts(results['records'], threshold=60.0)
        for alert in alerts_list:
            save_alert(
                detection_id=detection_id,
                attack_type=alert['attack_type'],
                confidence=alert['confidence'],
                source_ip=alert.get('source_ip'),
                severity=alert.get('severity', 'medium')
            )

        results['detection_id'] = detection_id
        results['model_used'] = best_model['model_name']
        results['alerts_generated'] = len(alerts_list)
        results['report_filename'] = report_filename

        return jsonify({'success': True, 'data': results})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Detection error: {str(e)}'}), 500


@app.route('/api/detection/history')
@login_required
def api_detection_history():
    """Retrieve historical detection batches."""
    history = get_detection_history(limit=50)
    return jsonify({'success': True, 'history': history})


@app.route('/api/reports/download/<int:detection_id>')
@login_required
def download_report(detection_id):
    """Serve CSV audit report for download."""
    history = get_detection_history(limit=100)
    target = next((item for item in history if item['id'] == detection_id), None)

    if target and target.get('report_path') and os.path.exists(target['report_path']):
        return send_file(target['report_path'], as_attachment=True)

    flash('The requested report is unavailable or has been archived.', 'warning')
    return redirect(url_for('detection'))


# ==========================================
# ATTACK ANALYSIS
# ==========================================

@app.route('/attack-analysis')
@login_required
def attack_analysis():
    """Visual cybersecurity threat distribution and timeline analysis."""
    stats = get_dashboard_stats()
    alerts = get_alerts(limit=200)
    history = get_detection_history(limit=50)
    return render_template('attack_analysis.html', stats=stats, alerts=alerts, history=history)


# ==========================================
# ALERTS SYSTEM
# ==========================================

@app.route('/alerts')
@login_required
def alerts_page():
    """Security alerts log and investigation center."""
    alerts = get_alerts(limit=100)
    return render_template('alerts.html', alerts=alerts)


@app.route('/api/alerts/data')
@login_required
def api_alerts_data():
    """JSON API for all system alerts."""
    unreviewed = request.args.get('unreviewed') == 'true'
    alerts = get_alerts(limit=100, unreviewed_only=unreviewed)
    return jsonify({'success': True, 'alerts': alerts})


@app.route('/api/alerts/review/<int:alert_id>', methods=['POST'])
@login_required
def api_alert_review(alert_id):
    """Mark specific alert as acknowledged by analyst."""
    try:
        mark_alert_reviewed(alert_id)
        return jsonify({'success': True, 'message': f'Alert #{alert_id} acknowledged.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/dashboard/stats')
@login_required
def api_dashboard_stats():
    """JSON API for dashboard graphs and metric counters."""
    stats = get_dashboard_stats()
    return jsonify({'success': True, 'stats': stats})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
