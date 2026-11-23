"""
Database access and operations module for IoT Intrusion Detection System.
Uses SQLite for persistent storage.
"""
import os
import sqlite3
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config


def get_db_path():
    """Ensure directory exists and return database path."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    return Config.DATABASE_PATH


def get_db():
    """Create a new database connection with row factory."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def close_db(conn=None):
    """Close connection if open."""
    if conn:
        conn.close()


def init_db():
    """Initialize database tables and create default admin user if none exists."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'admin',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trained_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        accuracy REAL NOT NULL,
        precision_score REAL NOT NULL,
        recall_score REAL NOT NULL,
        f1_score REAL NOT NULL,
        roc_auc REAL DEFAULT 0.0,
        training_time REAL NOT NULL,
        prediction_time REAL NOT NULL,
        model_path TEXT NOT NULL,
        is_best INTEGER DEFAULT 0,
        feature_names TEXT,
        label_classes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detection_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT NOT NULL,
        total_records INTEGER NOT NULL,
        normal_count INTEGER NOT NULL,
        malicious_count INTEGER NOT NULL,
        detection_rate REAL NOT NULL,
        model_used TEXT NOT NULL,
        report_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id INTEGER,
        attack_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        source_ip TEXT DEFAULT '192.168.1.50',
        severity TEXT DEFAULT 'medium',
        reviewed INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (detection_id) REFERENCES detection_results (id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        num_rows INTEGER NOT NULL,
        num_cols INTEGER NOT NULL,
        file_size REAL NOT NULL,
        columns TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preprocessing_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        missing_values INTEGER NOT NULL,
        duplicates_removed INTEGER NOT NULL,
        features_selected TEXT NOT NULL,
        scaling_method TEXT NOT NULL,
        label_column TEXT NOT NULL,
        num_classes INTEGER NOT NULL,
        class_names TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES dataset_info (id)
    );
    """)

    # Check for default admin user
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed = generate_password_hash("admin123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("admin", "admin@iot-ids.local", hashed, "admin")
        )

    conn.commit()
    conn.close()


def create_user(username, email, password, role="admin"):
    """Register a new user with hashed password."""
    conn = get_db()
    cursor = conn.cursor()
    hashed = generate_password_hash(password)
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username.strip(), email.strip().lower(), hashed, role)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    finally:
        conn.close()


def get_user_by_username(username):
    """Retrieve user record by username."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id):
    """Retrieve user record by primary key."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def verify_password(stored_hash, password):
    """Verify plain password against stored hash."""
    return check_password_hash(stored_hash, password)


def save_model_info(model_name, metrics_dict, model_path, feature_names=None, label_classes=None):
    """Insert trained model evaluation metrics into database."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        features_json = json.dumps(feature_names) if feature_names else "[]"
        labels_json = json.dumps(label_classes) if label_classes else "[]"

        cursor.execute("""
            INSERT INTO trained_models (
                model_name, accuracy, precision_score, recall_score, f1_score,
                roc_auc, training_time, prediction_time, model_path,
                feature_names, label_classes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            float(metrics_dict.get('accuracy', 0.0)),
            float(metrics_dict.get('precision', 0.0)),
            float(metrics_dict.get('recall', 0.0)),
            float(metrics_dict.get('f1_score', 0.0)),
            float(metrics_dict.get('roc_auc', 0.0)),
            float(metrics_dict.get('training_time', 0.0)),
            float(metrics_dict.get('prediction_time', 0.0)),
            model_path,
            features_json,
            labels_json
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def set_best_model(model_id):
    """Mark a model as the active best model and unmark others."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE trained_models SET is_best = 0")
        cursor.execute("UPDATE trained_models SET is_best = 1 WHERE id = ?", (model_id,))
        conn.commit()
    finally:
        conn.close()


def get_all_models():
    """Retrieve all trained models ordered by f1_score descending."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trained_models ORDER BY f1_score DESC, accuracy DESC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_best_model():
    """Get the currently designated best model or the top model by F1."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trained_models WHERE is_best = 1 LIMIT 1")
        row = cursor.fetchone()
        if not row:
            cursor.execute("SELECT * FROM trained_models ORDER BY f1_score DESC LIMIT 1")
            row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def save_detection_result(user_id, filename, total, normal, malicious, rate, model_name, report_path=None):
    """Store batch intrusion detection summary."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detection_results (
                user_id, filename, total_records, normal_count,
                malicious_count, detection_rate, model_used, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, filename, total, normal, malicious, rate, model_name, report_path))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_detection_history(user_id=None, limit=50):
    """Retrieve intrusion detection logs."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM detection_results WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit)
            )
        else:
            cursor.execute("SELECT * FROM detection_results ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_alert(detection_id, attack_type, confidence, source_ip=None, severity='medium'):
    """Save an individual intrusion alert."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (detection_id, attack_type, confidence, source_ip, severity)
            VALUES (?, ?, ?, ?, ?)
        """, (detection_id, attack_type, float(confidence), source_ip or '192.168.1.100', severity))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_alerts(limit=100, unreviewed_only=False):
    """Get alert notifications with optional filter."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        if unreviewed_only:
            cursor.execute("SELECT * FROM alerts WHERE reviewed = 0 ORDER BY created_at DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_alert_reviewed(alert_id):
    """Mark an alert as reviewed."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE alerts SET reviewed = 1 WHERE id = ?", (alert_id,))
        conn.commit()
    finally:
        conn.close()


def get_dashboard_stats():
    """Aggregate statistics for the dashboard cards and charts."""
    conn = get_db()
    try:
        cursor = conn.cursor()

        # Totals from detection_results
        cursor.execute("""
            SELECT 
                COALESCE(SUM(total_records), 0) AS total_records,
                COALESCE(SUM(normal_count), 0) AS total_normal,
                COALESCE(SUM(malicious_count), 0) AS total_malicious,
                COUNT(id) AS total_detections
            FROM detection_results
        """)
        totals = cursor.fetchone()

        total_records = totals['total_records']
        total_normal = totals['total_normal']
        total_malicious = totals['total_malicious']
        detection_rate = round((total_malicious / total_records * 100), 2) if total_records > 0 else 0.0

        # Total alerts count
        cursor.execute("SELECT COUNT(*) AS count FROM alerts")
        alerts_count = cursor.fetchone()['count']

        # Most frequent attack
        cursor.execute("""
            SELECT attack_type, COUNT(*) as count 
            FROM alerts 
            GROUP BY attack_type 
            ORDER BY count DESC 
            LIMIT 1
        """)
        top_attack_row = cursor.fetchone()
        most_frequent_attack = top_attack_row['attack_type'] if top_attack_row else "None Detected"

        # Best model accuracy
        cursor.execute("SELECT accuracy, model_name FROM trained_models WHERE is_best = 1 LIMIT 1")
        best_model = cursor.fetchone()
        best_accuracy = round(best_model['accuracy'] * 100, 2) if best_model else 0.0

        # Recent alerts (5)
        cursor.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 5")
        recent_alerts = [dict(r) for r in cursor.fetchall()]

        # Attack distribution counts
        cursor.execute("""
            SELECT attack_type, COUNT(*) as count 
            FROM alerts 
            GROUP BY attack_type 
            ORDER BY count DESC
        """)
        attack_dist = {r['attack_type']: r['count'] for r in cursor.fetchall()}

        return {
            'total_records': total_records,
            'total_normal': total_normal,
            'total_malicious': total_malicious,
            'detection_rate': detection_rate,
            'total_detections': totals['total_detections'],
            'total_alerts': alerts_count,
            'most_frequent_attack': most_frequent_attack,
            'best_accuracy': best_accuracy,
            'recent_alerts': recent_alerts,
            'attack_distribution': attack_dist
        }
    finally:
        conn.close()


def save_dataset_info(filename, num_rows, num_cols, file_size, columns):
    """Save metadata for an uploaded dataset."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cols_json = json.dumps(columns) if isinstance(columns, list) else str(columns)
        cursor.execute("""
            INSERT INTO dataset_info (filename, num_rows, num_cols, file_size, columns)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, num_rows, num_cols, round(file_size, 2), cols_json))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def save_preprocessing_info(dataset_id, missing_values, duplicates_removed, features_selected,
                            scaling_method, label_column, num_classes, class_names):
    """Store preprocessing summary."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        features_json = json.dumps(features_selected) if isinstance(features_selected, list) else str(features_selected)
        classes_json = json.dumps(class_names) if isinstance(class_names, list) else str(class_names)
        cursor.execute("""
            INSERT INTO preprocessing_info (
                dataset_id, missing_values, duplicates_removed, features_selected,
                scaling_method, label_column, num_classes, class_names
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (dataset_id, missing_values, duplicates_removed, features_json,
              scaling_method, label_column, num_classes, classes_json))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_latest_dataset_info():
    """Retrieve the most recently uploaded dataset record."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dataset_info ORDER BY uploaded_at DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            res = dict(row)
            try:
                res['columns'] = json.loads(res['columns'])
            except Exception:
                pass
            return res
        return None
    finally:
        conn.close()


def get_latest_preprocessing_info():
    """Retrieve the most recent preprocessing record."""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM preprocessing_info ORDER BY created_at DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            res = dict(row)
            try:
                res['features_selected'] = json.loads(res['features_selected'])
            except Exception:
                pass
            try:
                res['class_names'] = json.loads(res['class_names'])
            except Exception:
                pass
            return res
        return None
    finally:
        conn.close()
