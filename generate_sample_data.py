"""
Synthetic IoT Network Traffic Dataset Generator.
Generates realistic telemetry matching the CICIoT2023 feature schema for training and testing.
Supports both standard library (csv/random/math) and pandas/numpy.
"""
import os
import csv
import math
import random
from config import Config


def generate_dataset(output_path, num_samples=2000, seed=42):
    """Generate synthetic network flow dataset with Benign and diverse IoT attacks."""
    random.seed(seed)

    attack_types = [
        ('BenignTraffic', int(num_samples * 0.60)),
        ('DDoS-TCP_Flood', int(num_samples * 0.08)),
        ('DDoS-UDP_Flood', int(num_samples * 0.07)),
        ('DoS-HTTP_Flood', int(num_samples * 0.06)),
        ('Mirai-greeth_flood', int(num_samples * 0.05)),
        ('Recon-PortScan', int(num_samples * 0.05)),
        ('DDoS-SYN_Flood', int(num_samples * 0.04)),
        ('Recon-OSScan', int(num_samples * 0.03)),
        ('DictionaryBruteForce', int(num_samples * 0.02)),
    ]

    fieldnames = [
        'flow_duration', 'Header_Length', 'Protocol_Type', 'Duration',
        'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
        'ece_flag_number', 'cwr_flag_number', 'urg_flag_number',
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
        'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC',
        'Tot_sum', 'Min', 'Max', 'AVG', 'Std', 'Tot_size', 'IAT',
        'Number', 'Magnitude', 'Radius', 'Covariance', 'Variance',
        'Weight', 'label'
    ]

    rows = []

    for attack_name, count in attack_types:
        for _ in range(count):
            if attack_name == 'BenignTraffic':
                flow_duration = max(0.001, random.expovariate(1.0 / 15.0))
                header_length = random.randint(20, 60)
                protocol = 6 if random.random() < 0.7 else 17
                rate = max(1.0, random.gauss(120.0, 30.0))
                srate = rate * random.uniform(0.4, 0.6)
                drate = rate - srate
                syn = 1 if random.random() < 0.1 else 0
                ack = 1 if random.random() < 0.8 else 0
                fin = 1 if random.random() < 0.1 else 0
                rst = 0
                http = 1 if random.random() < 0.4 else 0
                https = 1 if random.random() < 0.5 else 0
                dns = 1 if random.random() < 0.2 else 0
                telnet = 0
                ssh = 1 if random.random() < 0.05 else 0
                tot_sum = max(10.0, random.gauss(5000, 1000))
                tot_size = max(40.0, random.gauss(1200, 300))
                iat = max(0.0001, random.expovariate(1.0 / 0.05))
                magnitude = max(1.0, random.gauss(10, 2))
                variance = max(0.1, random.uniform(5, 50))
            elif 'DDoS' in attack_name:
                flow_duration = max(0.001, random.expovariate(1.0 / 2.0))
                header_length = random.choice([20, 32, 40])
                protocol = 6 if 'TCP' in attack_name or 'SYN' in attack_name else 17
                rate = max(100.0, random.gauss(45000.0, 8000.0))
                srate = rate * random.uniform(0.85, 0.98)
                drate = rate - srate
                syn = 1 if 'SYN' in attack_name or random.random() < 0.8 else 0
                ack = 0 if 'SYN' in attack_name else (1 if random.random() < 0.2 else 0)
                fin = 0
                rst = 1 if random.random() < 0.4 else 0
                http = 0
                https = 0
                dns = 1 if 'UDP' in attack_name and random.random() < 0.5 else 0
                telnet = 0
                ssh = 0
                tot_sum = max(100.0, random.gauss(120000, 20000))
                tot_size = max(20.0, random.gauss(64, 15))
                iat = max(0.00001, random.expovariate(1.0 / 0.0001))
                magnitude = max(1.0, random.gauss(90, 15))
                variance = max(0.1, random.uniform(200, 1000))
            elif 'DoS' in attack_name:
                flow_duration = max(0.001, random.expovariate(1.0 / 5.0))
                header_length = random.randint(40, 80)
                protocol = 6
                rate = max(50.0, random.gauss(18000.0, 4000.0))
                srate = rate * random.uniform(0.7, 0.9)
                drate = rate - srate
                syn = 1 if random.random() < 0.5 else 0
                ack = 1 if random.random() < 0.5 else 0
                fin = 0
                rst = 0
                http = 1 if 'HTTP' in attack_name else 0
                https = 0
                dns = 0
                telnet = 0
                ssh = 0
                tot_sum = max(100.0, random.gauss(60000, 10000))
                tot_size = max(40.0, random.gauss(512, 100))
                iat = max(0.00001, random.expovariate(1.0 / 0.001))
                magnitude = max(1.0, random.gauss(50, 10))
                variance = max(0.1, random.uniform(80, 300))
            elif 'Mirai' in attack_name:
                flow_duration = max(0.001, random.expovariate(1.0 / 3.0))
                header_length = random.randint(30, 60)
                protocol = random.choice([47, 17])
                rate = max(50.0, random.gauss(28000.0, 5000.0))
                srate = rate * random.uniform(0.8, 0.95)
                drate = rate - srate
                syn = 0
                ack = 0
                fin = 0
                rst = 0
                http = 0
                https = 0
                dns = 0
                telnet = 1 if random.random() < 0.3 else 0
                ssh = 0
                tot_sum = max(100.0, random.gauss(85000, 15000))
                tot_size = max(40.0, random.gauss(128, 30))
                iat = max(0.00001, random.expovariate(1.0 / 0.0005))
                magnitude = max(1.0, random.gauss(70, 12))
                variance = max(0.1, random.uniform(150, 600))
            elif 'Recon' in attack_name:
                flow_duration = max(0.001, random.expovariate(1.0 / 0.8))
                header_length = 20
                protocol = 6
                rate = max(10.0, random.gauss(850.0, 150.0))
                srate = rate * random.uniform(0.9, 0.99)
                drate = rate - srate
                syn = 1
                ack = 0
                fin = 1 if random.random() < 0.2 else 0
                rst = 0
                http = 0
                https = 0
                dns = 0
                telnet = 1 if random.random() < 0.2 else 0
                ssh = 1 if random.random() < 0.2 else 0
                tot_sum = max(10.0, random.gauss(1200, 300))
                tot_size = 40.0
                iat = max(0.0001, random.expovariate(1.0 / 0.01))
                magnitude = max(1.0, random.gauss(25, 5))
                variance = max(0.1, random.uniform(10, 40))
            else:
                flow_duration = max(0.001, random.expovariate(1.0 / 10.0))
                header_length = 32
                protocol = 6
                rate = max(5.0, random.gauss(600.0, 100.0))
                srate = rate * 0.5
                drate = rate * 0.5
                syn = 1 if random.random() < 0.3 else 0
                ack = 1
                fin = 1 if random.random() < 0.3 else 0
                rst = 0
                http = 0
                https = 0
                dns = 0
                telnet = 1 if random.random() < 0.5 else 0
                ssh = 1 if not telnet else 0
                tot_sum = max(10.0, random.gauss(9000, 2000))
                tot_size = max(40.0, random.gauss(256, 50))
                iat = max(0.0001, random.expovariate(1.0 / 0.03))
                magnitude = max(1.0, random.gauss(30, 6))
                variance = max(0.1, random.uniform(15, 60))

            rows.append({
                'flow_duration': round(flow_duration, 4),
                'Header_Length': int(header_length),
                'Protocol_Type': int(protocol),
                'Duration': round(flow_duration * 0.8, 4),
                'Rate': round(rate, 2),
                'Srate': round(srate, 2),
                'Drate': round(drate, 2),
                'fin_flag_number': int(fin),
                'syn_flag_number': int(syn),
                'rst_flag_number': int(rst),
                'psh_flag_number': 1 if random.random() < 0.1 else 0,
                'ack_flag_number': int(ack),
                'ece_flag_number': 0,
                'cwr_flag_number': 0,
                'urg_flag_number': 0,
                'HTTP': int(http),
                'HTTPS': int(https),
                'DNS': int(dns),
                'Telnet': int(telnet),
                'SMTP': 0,
                'SSH': int(ssh),
                'IRC': 0,
                'TCP': 1 if protocol == 6 else 0,
                'UDP': 1 if protocol == 17 else 0,
                'DHCP': 0,
                'ARP': 0,
                'ICMP': 1 if protocol == 1 else 0,
                'IPv': 4,
                'LLC': 1,
                'Tot_sum': round(tot_sum, 2),
                'Min': round(tot_size * 0.5, 2),
                'Max': round(tot_size * 2.5, 2),
                'AVG': round(tot_size, 2),
                'Std': round(variance * 0.5, 2),
                'Tot_size': round(tot_size, 2),
                'IAT': round(iat, 6),
                'Number': random.randint(5, 500),
                'Magnitude': round(magnitude, 2),
                'Radius': round(magnitude * 0.3, 2),
                'Covariance': round(variance * 0.4, 2),
                'Variance': round(variance, 2),
                'Weight': round(random.uniform(1.0, 10.0), 2),
                'label': attack_name
            })

    random.shuffle(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} samples saved to: {output_path}")
    return rows


if __name__ == '__main__':
    train_file = os.path.join(Config.UPLOAD_FOLDER, 'sample_iot_traffic.csv')
    test_file = os.path.join(Config.UPLOAD_FOLDER, 'test_traffic.csv')

    print("Generating training dataset (2000 samples)...")
    generate_dataset(train_file, num_samples=2000, seed=42)

    print("\nGenerating test inference dataset (250 samples)...")
    generate_dataset(test_file, num_samples=250, seed=123)
