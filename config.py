"""
Configuration settings for IoT Intrusion Detection System.
"""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'iot-ids-cybersecurity-secret-key-2026')
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'iot_ids.db')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'datasets')
    MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
    REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload
    ALLOWED_EXTENSIONS = {'csv'}

    # ML Configuration
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    MAX_FEATURES = 25

    # Supported model algorithms
    MODEL_NAMES = [
        'Random Forest',
        'Decision Tree',
        'Support Vector Machine',
        'K-Nearest Neighbors',
        'XGBoost'
    ]

    # Attack category mappings (CICIoT2023 / BoT-IoT aligned)
    ATTACK_CATEGORIES = {
        'BenignTraffic': 'Normal',
        'Normal': 'Normal',
        'DDoS-ICMP_Flood': 'DDoS',
        'DDoS-UDP_Flood': 'DDoS',
        'DDoS-TCP_Flood': 'DDoS',
        'DDoS-PSHACK_Flood': 'DDoS',
        'DDoS-SYN_Flood': 'DDoS',
        'DDoS-RSTFINFlood': 'DDoS',
        'DDoS-SynonymousIP_Flood': 'DDoS',
        'DoS-UDP_Flood': 'DoS',
        'DoS-TCP_Flood': 'DoS',
        'DoS-HTTP_Flood': 'DoS',
        'Mirai-greeth_flood': 'Mirai Botnet',
        'Mirai-greip_flood': 'Mirai Botnet',
        'Mirai-udpplain': 'Mirai Botnet',
        'Recon-PingSweep': 'Reconnaissance',
        'Recon-OSScan': 'Reconnaissance',
        'Recon-PortScan': 'Reconnaissance',
        'Recon-HostDiscovery': 'Reconnaissance',
        'BrowserHijacking': 'Theft/Hijacking',
        'CommandInjection': 'Injection',
        'SqlInjection': 'Injection',
        'XSS': 'Injection',
        'Backdoor_Malware': 'Malware',
        'Uploading_Attack': 'Malware/Theft',
        'DictionaryBruteForce': 'Brute Force',
        'MITM-ArpSpoofing': 'Spoofing',
        'DNS_Spoofing': 'Spoofing',
        'VulnerabilityScan': 'Reconnaissance',
    }
