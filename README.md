# Intelligent Intrusion Detection System for IoT Networks Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%26%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Cybersecurity](https://img.shields.io/badge/Cybersecurity-IoT%20NIDS-red.svg)](#)

An intelligent, full-stack, enterprise-grade Network Intrusion Detection System (NIDS) specifically tailored for Internet of Things (IoT) ecosystems. Powered by state-of-the-art Machine Learning algorithms, this platform analyzes telemetry traffic, detects malicious signatures, identifies threat vectors, mitigates data leakage, and provides security analysts with an interactive real-time command center.

---

## Table of Contents
- [1. System Architecture](#1-system-architecture)
- [2. Core Features](#2-core-features)
- [3. Machine Learning Algorithms](#3-machine-learning-algorithms)
- [4. Project Structure](#4-project-structure)
- [5. Installation & Setup](#5-installation--setup)
- [6. Synthetic Dataset & Benchmarks](#6-synthetic-dataset--benchmarks)
- [7. Complete Workflow Guide](#7-complete-workflow-guide)
- [8. REST API Endpoints](#8-rest-api-endpoints)
- [9. Automated Testing](#9-automated-testing)
- [10. License](#10-license)

---

## 1. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                           IoT Network Telemetry Layer                            |
|             (Sensors, Gateways, MQTT, HTTP, DNS, TCP/UDP Telemetry)               |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                        Dataset Ingestion & Validation                             |
|       - CSV Schema Validation     - Inf/NaN Detection    - Redundancy Pruning     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                 Leakage-Free Preprocessing Pipeline (Joblib)                      |
|       - ANOVA/Variance Selection  - Target Label Encoding  - StandardScaler       |
|       - Stratified Train-Test Split                      - SMOTE (Train-Only)     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     Multi-Model Training & Benchmark Suite                        |
|   ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐      |
|   │  Random Forest  │  Decision Tree  │       SVM       │       KNN       │      |
|   └─────────────────┴─────────────────┴─────────────────┴─────────────────┘      |
|                                       │                                           |
|                            ┌──────────────────┐                                   |
|                            │     XGBoost      │                                   |
|                            └──────────────────┘                                   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                  Automated Model Evaluation & Best Selection                      |
|    - Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC (OVR)       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                   Real-Time Batch Detection & Alerting Engine                     |
|       - Threat Classification     - Confidence Scoring    - Severity Rating       |
|       - Audit Report (.CSV)       - Automated Alert Log   - Analyst Review Gate   |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                         Cybersecurity Command Center                              |
|           (Dark Navy Glassmorphic UI, Chart.js, Plotly, Responsive Dashboard)     |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Features

- **Multi-Algorithm Model Training**: Trains and benchmarks 5 classical and ensemble algorithms side by side.
- **Leakage-Free Pipeline**: Scaling parameters and synthetic oversampling (SMOTE) are strictly isolated to the training partition.
- **Automated Feature Selection**: Combines variance filtering and SelectKBest (ANOVA F-statistic) to isolate the most salient traffic features.
- **Real-Time Batch Detection**: Upload live traffic batches and receive instant classification (Normal vs Attack) with confidence percentages.
- **Taxonomy of IoT Threats**: Identifies DDoS (SYN/UDP/TCP/ICMP), DoS, Mirai Botnet, Reconnaissance (PortScan/OSScan), and Brute Force attacks.
- **Incident Alert Management**: Real-time alert feed with severity tiering (Low, Medium, High) and review/acknowledgment states.
- **Downloadable Forensic Reports**: Generates downloadable timestamped audit CSVs for security auditing.
- **Secure Authentication**: Salted password hashing with Werkzeug and session-guarded route decorators.

---

## 3. Machine Learning Algorithms

| Algorithm | Method & Strengths | Optimized Metric |
|-----------|--------------------|------------------|
| **Random Forest** | 100 bagged estimators; robust to noise and outliers | High F1 & Recall |
| **Decision Tree** | Interpretable non-linear split boundaries; sub-millisecond inference | Inference Latency |
| **Support Vector Machine (SVM)** | RBF kernel maximum-margin separation in high dimensions | High Precision |
| **K-Nearest Neighbors (KNN)** | Instance-based local clustering in scaled metric space | Multi-class Detection |
| **XGBoost Classifier** | Scaled gradient boosted decision trees with regularized learning | Overall Top Accuracy |

---

## 4. Project Structure

```text
cloned_ids/
├── app.py                      # Core Flask web application and routing controller
├── config.py                   # Central configuration (Paths, Hyperparameters, Attacks)
├── requirements.txt            # Python dependencies
├── generate_sample_data.py     # Standalone synthetic IoT telemetry generator
├── README.md                   # System documentation
│
├── database/                   # SQLite database operations
│   ├── __init__.py
│   └── db.py                   # Schema creation, queries, user & alert handling
│
├── preprocessing/              # Data preprocessing module
│   ├── __init__.py
│   └── preprocessor.py         # Cleaning, encoding, feature selection, scaling
│
├── training/                   # Model training orchestrator
│   ├── __init__.py
│   └── trainer.py              # Parallel algorithm training and timing
│
├── evaluation/                 # Metrics & evaluation suite
│   ├── __init__.py
│   └── evaluator.py            # Confusion matrix, ROC-AUC, classification reports
│
├── detection/                  # Real-time inference engine
│   ├── __init__.py
│   └── detector.py             # Batch prediction, alert trigger, report generation
│
├── templates/                  # Jinja2 HTML5 UI views (Dark Navy Cyber Theme)
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── dataset.html
│   ├── preprocessing.html
│   ├── training.html
│   ├── model_comparison.html
│   ├── detection.html
│   ├── attack_analysis.html
│   └── alerts.html
│
├── static/                     # Frontend stylesheets and scripts
│   ├── css/
│   │   └── style.css           # Glassmorphism dark navy cybersecurity design
│   └── js/
│       └── main.js             # AJAX caller, charts, counters, notifications
│
├── datasets/                   # Datasets directory (Raw & Sample CSVs)
├── models/                     # Saved joblib pipelines and trained models
├── reports/                    # Exported audit CSV reports
└── tests/                      # Automated test suite
    ├── __init__.py
    └── test_app.py             # 25+ comprehensive automated tests
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.9+ or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/thanush0001/IDS.git
cd IDS
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Synthetic Datasets
```bash
python generate_sample_data.py
```
This command generates:
- `datasets/sample_iot_traffic.csv` (2,000 samples for training)
- `datasets/test_traffic.csv` (250 samples for batch inference testing)

### 5. Start the Web Server
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

**Default Administrator Credentials:**
- **Username**: `admin`
- **Password**: `admin123`

---

## 6. Synthetic Dataset & Benchmarks

The built-in telemetry generator mimics the feature taxonomy of the **CICIoT2023** benchmark:
- **Flow Metrics**: `flow_duration`, `Rate`, `Srate`, `Drate`, `IAT`, `Tot_sum`, `Tot_size`
- **Protocol Flags**: `syn_flag_number`, `ack_flag_number`, `rst_flag_number`, `fin_flag_number`
- **Application Protocols**: `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SSH`, `TCP`, `UDP`, `ICMP`
- **Attack Classes**: `BenignTraffic`, `DDoS-TCP_Flood`, `DDoS-UDP_Flood`, `DoS-HTTP_Flood`, `Mirai-greeth_flood`, `Recon-PortScan`, `DDoS-SYN_Flood`, `Recon-OSScan`, `DictionaryBruteForce`

---

## 7. Complete Workflow Guide

1. **Authenticate**: Log in with administrator credentials.
2. **Dataset Manager**: Upload a custom CSV or register `datasets/sample_iot_traffic.csv`. View instant dataset profiling.
3. **Preprocessing**: Click **"Execute Preprocessing Pipeline"** to run imputation, encoding, feature selection, and train-test splitting.
4. **Model Training**: Click **"Train All 5 Algorithms"** to fit Random Forest, Decision Tree, SVM, KNN, and XGBoost.
5. **Model Comparison**: Review comparative charts (Accuracy, F1-Score, Training Speed) and inspect the designated Best Model.
6. **Intrusion Detection**: Upload `datasets/test_traffic.csv` to run real-time inference, generate severity-rated alerts, and download the full forensic audit CSV report.
7. **Security Alerts**: Review and acknowledge active security incident tickets.

---

## 8. REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/stats` | `GET` | Aggregated telemetry counts, attack breakdown, detection rate |
| `/api/dataset/preview` | `GET` | Head preview (first 20 rows) of active dataset |
| `/api/dataset/info` | `GET` | Detailed column statistics and missing value summary |
| `/api/preprocessing/run` | `POST` | Executes data cleaning, feature extraction, and scaling |
| `/api/training/run` | `POST` | Trains and evaluates all 5 ML classifiers |
| `/api/models/comparison` | `GET` | Performance metrics for all trained models |
| `/api/detection/predict` | `POST` | Multipart upload for batch intrusion classification |
| `/api/detection/history` | `GET` | History of all detection scans |
| `/api/alerts/data` | `GET` | List all system alerts (with unreviewed filter) |
| `/api/alerts/review/<id>` | `POST` | Marks an alert as acknowledged |
| `/api/reports/download/<id>` | `GET` | Downloads audit report CSV |

---

## 9. Automated Testing

Run the comprehensive test suite with `pytest`:

```bash
pytest -v tests/test_app.py
```

The test suite validates:
- SQLite schema generation and user credential hashing
- Full preprocessing pipeline and transformer serialization
- Training and evaluation across all 5 ML algorithms
- Real-time batch detection and alert generation
- Authentication security and web route permissions

---

## 10. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
