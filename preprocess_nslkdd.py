"""
NSL-KDD Dataset Preprocessing Script

This script:
1. Loads NSL-KDD data from raw files
2. Adds proper column names
3. Performs feature selection using Mutual Information (top 25%)
4. Saves train/test splits in the format expected by CausalTGAN
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# NSL-KDD column names (41 features + 1 label + 1 difficulty)
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


def load_nslkdd_data(file_path):
    """Load NSL-KDD data and add column names"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def preprocess_labels(df):
    """
    Preprocess attack labels:
    - Convert detailed attack names to binary or multi-class
    - Option 1: Binary (normal vs attack)
    - Option 2: Keep main attack categories
    """
    # Create a copy of label column
    df['label_original'] = df['label'].copy()

    # Map detailed attacks to main categories
    attack_mapping = {
        'normal': 'normal',
        # DoS attacks
        'neptune': 'dos', 'smurf': 'dos', 'pod': 'dos', 'teardrop': 'dos',
        'land': 'dos', 'back': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
        'processtable': 'dos', 'mailbomb': 'dos',
        # Probe attacks
        'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
        'mscan': 'probe', 'saint': 'probe',
        # R2L attacks
        'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
        'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l', 'spy': 'r2l',
        'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
        'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
        # U2R attacks
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 'perl': 'u2r',
        'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r'
    }

    df['label'] = df['label'].map(attack_mapping)

    # Handle any unmapped labels (shouldn't happen, but just in case)
    if df['label'].isna().any():
        print(f"Warning: {df['label'].isna().sum()} unmapped labels found!")
        print("Unmapped labels:", df[df['label'].isna()]['label_original'].unique())
        # Fill with 'unknown'
        df['label'] = df['label'].fillna('unknown')

    print("\nLabel distribution:")
    print(df['label'].value_counts())

    return df


def select_features_mutual_info(X, y, percentile=75):
    """
    Select features using Mutual Information
    Keep top 'percentile'% of features
    """
    print(f"\nPerforming feature selection (top {percentile}%)...")

    # Encode categorical features for MI calculation
    X_encoded = X.copy()
    label_encoders = {}

    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Calculate MI scores
    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X.columns)

    # Select top features
    threshold = np.percentile(mi_scores, percentile)
    selected_features = mi_scores[mi_scores >= threshold].sort_values(ascending=False)

    print(f"\nTop features (MI scores):")
    for feat, score in selected_features.head(15).items():
        print(f"  {feat:30s}: {score:.4f}")

    print(f"\nSelected {len(selected_features)} features out of {len(X.columns)}")

    return selected_features.index.tolist(), mi_scores


def main():
    # Configuration
    RAW_TRAIN_PATH = r"C:\Users\qadee\Downloads\KDDTrain+_20Percent.txt\KDDTrain+_20Percent.txt"
    OUTPUT_DIR = r"C:\Users\qadee\Desktop\CausalTGAN\data\real_world\nsl_kdd"
    FEATURE_SELECTION_PERCENTILE = 75  # Keep top 25%

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_nslkdd_data(RAW_TRAIN_PATH)

    # Remove difficulty column (not needed for training)
    df = df.drop('difficulty', axis=1)

    # Preprocess labels
    df = preprocess_labels(df)

    # Separate features and labels
    X = df.drop(['label', 'label_original'], axis=1)
    y = df['label']

    # Feature selection
    selected_features, mi_scores = select_features_mutual_info(X, y, FEATURE_SELECTION_PERCENTILE)

    # Create final dataset with selected features
    X_selected = X[selected_features]
    train_df = pd.concat([X_selected, y], axis=1)

    print(f"\nTrain set: {len(train_df)} samples")
    print("Test set will be added separately")

    # Save to CSV (with index for compatibility with CausalTGAN loading)
    train_path = os.path.join(OUTPUT_DIR, 'train.csv')
    train_df.to_csv(train_path, index=True)

    print(f"\nSaved train data to: {train_path}")

    # Create placeholder test.csv (will be replaced with actual test data)
    test_path = os.path.join(OUTPUT_DIR, 'test.csv')
    # Save a small sample as placeholder
    train_df.head(100).to_csv(test_path, index=True)
    print(f"Created placeholder test.csv (replace with actual test data later)")

    # Save feature information
    feature_info = {
        'selected_features': selected_features,
        'mi_scores': mi_scores.to_dict(),
        'num_features': len(selected_features),
        'categorical_features': [col for col in selected_features if col in
                                 ['protocol_type', 'service', 'flag', 'land',
                                  'logged_in', 'root_shell', 'su_attempted',
                                  'is_host_login', 'is_guest_login']]
    }

    import pickle
    with open(os.path.join(OUTPUT_DIR, 'feature_info.pkl'), 'wb') as f:
        pickle.dump(feature_info, f)

    print(f"\nSaved feature info to: {os.path.join(OUTPUT_DIR, 'feature_info.pkl')}")

    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Training samples: {len(train_df)}")
    print(f"Total features: {len(X.columns)} -> {len(selected_features)}")
    print(f"Categorical features: {len(feature_info['categorical_features'])}")
    print(f"Continuous features: {len(selected_features) - len(feature_info['categorical_features'])}")
    print(f"\nCategorical features: {feature_info['categorical_features']}")
    print("\nNext steps:")
    print("1. Add test set file (KDDTest+.txt) when available")
    print("2. Create causal graph (we'll do this next)")
    print("3. Update helper/constant.py with NSL_KDD_CATEGORY")
    print("4. Update helper/utils.py to recognize 'nsl_kdd' dataset")
    print("5. Run training: python train.py --data_name nsl_kdd")


if __name__ == '__main__':
    main()
