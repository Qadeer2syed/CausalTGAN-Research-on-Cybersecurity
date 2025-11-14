"""
Evaluate Synthetic NSL-KDD Data Quality
Compare synthetic data with real data using multiple metrics
Train on Synthetic/Real, Test on Official NSL-KDD Test Set
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Load training, test, and synthetic data"""
    print("\n[Step 1] Loading data...")

    # Load real training data (full 42 features)
    train_data = pd.read_csv('./data/real_world/nsl_kdd/train_full.csv').iloc[:, 1:]
    print(f"  Training data: {len(train_data)} samples, {len(train_data.columns)} features")

    # Load real test data (official NSL-KDD test set)
    test_data = pd.read_csv('./data/real_world/nsl_kdd/test_full.csv').iloc[:, 1:]
    print(f"  Test data: {len(test_data)} samples, {len(test_data.columns)} features")

    # Load synthetic data (partial knowledge model - all 42 features)
    synthetic_data = pd.read_csv('./Testing/CausalTGAN_runs_new_3_nsl_kdd2025.11.13--13-57-06/generated_samples.csv')
    print(f"  Synthetic data: {len(synthetic_data)} samples, {len(synthetic_data.columns)} features")

    return train_data, test_data, synthetic_data

def compare_distributions(train_data, test_data, synthetic_data):
    """Compare feature distributions"""
    print("\n[Step 2] Comparing feature distributions...")

    # Label distribution
    print("\n  Label Distribution Comparison:")

    print("\n  Training Data:")
    train_labels = train_data['label'].value_counts()
    for label, count in train_labels.items():
        print(f"    {label:10s}: {count:6d} ({count/len(train_data)*100:5.2f}%)")

    print("\n  Test Data:")
    test_labels = test_data['label'].value_counts()
    for label, count in test_labels.items():
        print(f"    {label:10s}: {count:6d} ({count/len(test_data)*100:5.2f}%)")

    print("\n  Synthetic Data:")
    synthetic_labels = synthetic_data['label'].value_counts()
    for label, count in synthetic_labels.items():
        print(f"    {label:10s}: {count:6d} ({count/len(synthetic_data)*100:5.2f}%)")

    # Statistical comparison for continuous features (10 CRFS causal features)
    print("\n  CRFS Causal Feature Statistics:")
    continuous_features = ['src_bytes', 'dst_bytes', 'diff_srv_rate',
                          'dst_host_srv_count', 'dst_host_same_srv_rate',
                          'dst_host_diff_srv_rate', 'dst_host_serror_rate', 'count']
    print("  (Showing 10 CRFS causal features identified by Algorithm 1)")

    print(f"  {'Feature':<25s} {'Train Mean':<15s} {'Synth Mean':<15s} {'Diff %':<10s}")
    print("  " + "-"*65)

    for feature in continuous_features:
        if feature in train_data.columns:
            train_mean = train_data[feature].mean()
            synth_mean = synthetic_data[feature].mean()
            diff_pct = abs(train_mean - synth_mean) / (train_mean + 1e-10) * 100
            print(f"  {feature:<25s} {train_mean:<15.4f} {synth_mean:<15.4f} {diff_pct:<10.2f}")

def train_on_synthetic_test_on_testset(train_data, test_data, synthetic_data):
    """Train classifier on synthetic data, test on official test set"""
    print("\n[Step 3] Train on Synthetic, Test on Official Test Set...")
    print("  (This tests if synthetic data can replace real training data)")

    # Encode categorical features
    categorical_features = ['protocol_type', 'service', 'flag']

    test_encoded = test_data.copy()
    synthetic_encoded = synthetic_data.copy()

    for col in categorical_features:
        le_col = LabelEncoder()
        # Fit on combined data to ensure same encoding
        combined = pd.concat([train_data[col], test_data[col], synthetic_data[col]])
        le_col.fit(combined)
        test_encoded[col] = le_col.transform(test_data[col])
        synthetic_encoded[col] = le_col.transform(synthetic_data[col])

    # Prepare features and labels
    X_train = synthetic_encoded.drop('label', axis=1)
    y_train = synthetic_encoded['label']

    X_test = test_encoded.drop('label', axis=1)
    y_test = test_encoded['label']

    # Train Random Forest
    print("  Training Random Forest classifier on synthetic data...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy on Test Set: {accuracy*100:.2f}%")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return accuracy

def train_on_real_test_on_testset(train_data, test_data):
    """Baseline: Train on real training data, test on official test set"""
    print("\n[Step 4] Train on Real, Test on Official Test Set (Baseline)...")

    # Encode categorical features
    categorical_features = ['protocol_type', 'service', 'flag']

    train_encoded = train_data.copy()
    test_encoded = test_data.copy()

    for col in categorical_features:
        le_col = LabelEncoder()
        # Fit on combined data
        combined = pd.concat([train_data[col], test_data[col]])
        le_col.fit(combined)
        train_encoded[col] = le_col.transform(train_data[col])
        test_encoded[col] = le_col.transform(test_data[col])

    # Prepare features and labels
    X_train = train_encoded.drop('label', axis=1)
    y_train = train_encoded['label']

    X_test = test_encoded.drop('label', axis=1)
    y_test = test_encoded['label']

    # Train
    print("  Training Random Forest classifier on real training data...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy on Test Set: {accuracy*100:.2f}%")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return accuracy

def main():
    print("="*70)
    print("EVALUATING SYNTHETIC NSL-KDD DATA QUALITY")
    print("="*70)
    print("Evaluation Protocol:")
    print("  - Train on Synthetic → Test on Official NSL-KDD Test Set")
    print("  - Train on Real → Test on Official NSL-KDD Test Set (Baseline)")
    print("  - Calculate Quality Ratio")
    print("="*70)

    # Load data
    train_data, test_data, synthetic_data = load_data()

    # Compare distributions
    compare_distributions(train_data, test_data, synthetic_data)

    # Train on synthetic, test on test set
    synth_accuracy = train_on_synthetic_test_on_testset(train_data, test_data, synthetic_data)

    # Baseline: train on real, test on test set
    real_accuracy = train_on_real_test_on_testset(train_data, test_data)

    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Train on Synthetic → Test on Test Set: {synth_accuracy*100:.2f}%")
    print(f"Train on Real → Test on Test Set:      {real_accuracy*100:.2f}%")
    print(f"Quality Ratio:                         {synth_accuracy/real_accuracy*100:.2f}%")
    print("\nInterpretation:")
    if synth_accuracy/real_accuracy > 0.95:
        print("  [EXCELLENT] Synthetic data captures >95% of real data utility")
    elif synth_accuracy/real_accuracy > 0.90:
        print("  [VERY GOOD] Synthetic data captures >90% of real data utility")
    elif synth_accuracy/real_accuracy > 0.80:
        print("  [GOOD] Synthetic data captures >80% of real data utility")
    elif synth_accuracy/real_accuracy > 0.70:
        print("  [FAIR] Synthetic data captures >70% of real data patterns")
    else:
        print("  [NEEDS IMPROVEMENT] Synthetic data quality could be improved")

    print("\nNotes:")
    print("  - Zeng et al. (2021): 99.33% accuracy on NSL-KDD using CRFS features + traditional ML")
    print("  - Wen et al. (2022): Causal-TGAN outperformed baselines on 6 real-world datasets")
    print("  - This work: Combining CRFS (10 causal features) + Causal-TGAN (hybrid mode) for NSL-KDD")
    print("="*70)

if __name__ == '__main__':
    main()
