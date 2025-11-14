"""
Preprocess Full NSL-KDD Dataset (All 41 Features)
For Partial Knowledge Training with Causal TGAN

This script:
1. Loads the complete NSL-KDD dataset (all 41 features)
2. Preprocesses labels (maps detailed attacks to 5 categories)
3. Saves the full dataset for training

The causal graph (10 CRFS features) stays separate, enabling hybrid mode:
- Causal generator: 10 CRFS features
- Conditional GAN: Remaining 31 features
"""

import pandas as pd
import numpy as np
import os

# NSL-KDD column names (41 features)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Attack type mapping (from detailed to 5 categories)
ATTACK_MAPPING = {
    'normal': 'normal',
    # DoS attacks
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos',
    'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos',
    'udpstorm': 'dos',
    # Probe attacks
    'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    # R2L (Remote to Local) attacks
    'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
    'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l', 'spy': 'r2l',
    'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
    'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    # U2R (User to Root) attacks
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 'perl': 'u2r',
    'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
}

def load_nslkdd_data(filepath):
    """Load NSL-KDD data with all features"""
    print(f"Loading data from: {filepath}")

    # Load data
    data = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)

    print(f"  Loaded: {len(data)} samples, {len(data.columns)} columns")

    # Remove difficulty column (last column, not a feature)
    data = data.drop('difficulty', axis=1)

    return data

def preprocess_labels(data):
    """Map detailed attack labels to 5 categories"""
    print("\nPreprocessing labels...")

    # Remove trailing dot from labels (e.g., "normal." -> "normal")
    data['label'] = data['label'].str.rstrip('.')

    # Map to 5 categories
    data['label'] = data['label'].map(ATTACK_MAPPING)

    # Check for any unmapped labels
    unmapped = data[data['label'].isna()]
    if len(unmapped) > 0:
        print(f"  [WARNING] {len(unmapped)} samples with unmapped labels")
        print(f"  Unmapped labels: {unmapped['label'].unique()}")
        # Drop unmapped samples
        data = data.dropna(subset=['label'])

    # Label distribution
    print("\n  Label distribution:")
    label_counts = data['label'].value_counts()
    for label, count in label_counts.items():
        print(f"    {label:10s}: {count:6d} ({count/len(data)*100:5.2f}%)")

    return data

def save_data(data, output_dir):
    """Save preprocessed data"""
    os.makedirs(output_dir, exist_ok=True)

    # Save full training data
    train_path = os.path.join(output_dir, 'train_full.csv')
    data.to_csv(train_path, index=True)
    print(f"\n[OK] Saved full dataset: {train_path}")
    print(f"     {len(data)} samples, {len(data.columns)} features")

    return train_path

def main():
    INPUT_FILE = r"C:\Users\qadee\Downloads\KDDTrain+_20Percent.txt\KDDTrain+_20Percent.txt"
    OUTPUT_DIR = r"C:\Users\qadee\Desktop\CausalTGAN\data\real_world\nsl_kdd"

    print("="*70)
    print("PREPROCESSING FULL NSL-KDD DATASET (ALL 41 FEATURES)")
    print("="*70)
    print("Purpose: Enable Partial Knowledge Training")
    print("  - Causal Generator: 10 CRFS features")
    print("  - Conditional GAN: Remaining 31 features")
    print("="*70)

    # Load data
    data = load_nslkdd_data(INPUT_FILE)

    # Preprocess labels
    data = preprocess_labels(data)

    # Save
    output_path = save_data(data, OUTPUT_DIR)

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"\nFeatures: {list(data.columns)}")
    print(f"\nNext steps:")
    print("  1. Update helper/utils.py to load train_full.csv")
    print("  2. Train: python train.py --data_name nsl_kdd --epochs 400 --batch_size 500")
    print("  3. This will trigger PARTIAL KNOWLEDGE (hybrid) mode automatically")
    print("="*70)

if __name__ == '__main__':
    main()
