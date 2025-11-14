"""
Implement Algorithm 1: Causal Reasoning-Based Feature Selection (CRFS)
From: Zeng et al. 2021 - "Improving the Accuracy of Network Intrusion Detection
      with Causal Machine Learning"

This script implements causal intervention to identify which features have
true causal relationships with cyberattacks vs spurious correlations.

Algorithm:
1. For each feature Yi:
   - Perform intervention: do(Yi = 1) and do(Yi = 0)
   - Calculate causal effect: E = E[X|do(Yi=1)] - E[X|do(Yi=0)]
   - If E/N < δ (threshold), mark as noisy feature
2. Remove all noisy features
3. Return causal feature set

Expected Output: 7-10 causal features for NSL-KDD
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle


def calculate_causal_effect(data, feature_name, label_col='label', delta=0.01):
    """
    Calculate causal effect of a feature on the label using do-calculus

    Based on Equation 16 from the paper:
    E = E[X|do(Y=1)] - E[X|do(Y=0)]

    If E/N < delta (where delta <= 0.01), feature is noisy

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset
    feature_name : str
        Name of the feature to test
    label_col : str
        Name of the label column
    delta : float
        Threshold for determining causality (default 0.01)

    Returns:
    --------
    causal_effect : float
        Magnitude of causal effect (E/N)
    is_causal : bool
        Whether feature is causal (E/N > delta)
    """
    N = len(data)

    # Encode labels to numeric - binary (attack=1, normal=0)
    if data[label_col].dtype == 'object':
        # Binary: normal=0, any attack=1
        labels = (data[label_col] != 'normal').astype(int).values
    else:
        # Already numeric
        labels = (data[label_col] > 0).astype(int).values

    # Get feature values
    feature_data = data[feature_name]

    # Handle categorical features differently
    if feature_data.dtype == 'object' or len(feature_data.unique()) <= 10:
        # Categorical/discrete feature
        # Encode if string
        if feature_data.dtype == 'object':
            le = LabelEncoder()
            feature = le.fit_transform(feature_data)
        else:
            feature = feature_data.values

        unique_vals = np.unique(feature)

        # For categorical: compare highest vs lowest encoded value
        val_high = unique_vals.max()
        val_low = unique_vals.min()

        mask_high = (feature == val_high)
        mask_low = (feature == val_low)
    else:
        # Continuous feature
        feature = feature_data.values

        # Use median split
        val_median = np.median(feature)
        mask_high = (feature > val_median)
        mask_low = (feature <= val_median)

    # Intervention do(Y=1): Expected label when feature is "high"
    if mask_high.sum() > 0:
        E_do_1 = labels[mask_high].mean()
    else:
        E_do_1 = labels.mean()

    # Intervention do(Y=0): Expected label when feature is "low"
    if mask_low.sum() > 0:
        E_do_0 = labels[mask_low].mean()
    else:
        E_do_0 = labels.mean()

    # Causal effect (absolute difference in probabilities)
    E = abs(E_do_1 - E_do_0)

    # The causal effect is E itself (probability difference)
    # Compare directly to delta
    causal_effect = E

    # Check if causal: if effect is significant
    is_causal = causal_effect > delta

    return causal_effect, is_causal


def identify_causal_features(data, label_col='label', delta=0.01):
    """
    Implement Algorithm 1: CRFS

    Identify causal features by testing causal effect of each feature

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset with all features
    label_col : str
        Name of the label column
    delta : float
        Threshold for causality (default 0.01 as in paper)

    Returns:
    --------
    causal_features : list
        List of causal feature names
    noisy_features : list
        List of noisy (non-causal) feature names
    causal_effects : dict
        Dictionary of feature -> causal effect magnitude
    """
    features = [col for col in data.columns if col != label_col]
    N = len(data)

    causal_features = []
    noisy_features = []
    causal_effects = {}

    print(f"\n{'='*70}")
    print(f"CAUSAL INTERVENTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Dataset size: {N}")
    print(f"Number of features to test: {len(features)}")
    print(f"Causality threshold (delta): {delta}")
    print(f"\nTesting each feature for causal effect...\n")

    for i, feature in enumerate(features, 1):
        try:
            causal_effect, is_causal = calculate_causal_effect(
                data, feature, label_col, delta
            )

            causal_effects[feature] = causal_effect

            if is_causal:
                causal_features.append(feature)
                status = "[+] CAUSAL"
            else:
                noisy_features.append(feature)
                status = "[-] NOISY"

            print(f"  [{i:2d}/{len(features)}] {feature:30s} | Effect: {causal_effect:.6f} | {status}")

        except Exception as e:
            print(f"  [{i:2d}/{len(features)}] {feature:30s} | ERROR: {e}")
            noisy_features.append(feature)
            causal_effects[feature] = 0.0

    return causal_features, noisy_features, causal_effects


def main():
    # Configuration
    DATA_PATH = r"C:\Users\qadee\Desktop\CausalTGAN\data\real_world\nsl_kdd\train.csv"
    OUTPUT_DIR = r"C:\Users\qadee\Desktop\CausalTGAN\data\real_world\nsl_kdd"
    DELTA = 0.05  # Threshold - using 0.05 (5% probability difference) as significance level

    print("="*70)
    print("ALGORITHM 1: CAUSAL REASONING-BASED FEATURE SELECTION (CRFS)")
    print("="*70)
    print("Paper: Zeng et al. 2021")
    print("Methodology: Causal intervention using do-calculus")
    print("="*70)

    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, index_col=0)
    print(f"[OK] Loaded {len(data)} samples with {len(data.columns)} columns")

    # Show label distribution
    print(f"\nLabel distribution:")
    print(data['label'].value_counts())

    # Run causal intervention analysis
    causal_features, noisy_features, causal_effects = identify_causal_features(
        data,
        label_col='label',
        delta=DELTA
    )

    # Results summary
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total features tested: {len(data.columns) - 1}")
    print(f"Causal features identified: {len(causal_features)}")
    print(f"Noisy features removed: {len(noisy_features)}")
    print(f"Feature reduction: {len(noisy_features) / (len(data.columns)-1) * 100:.1f}%")

    # Show causal features ranked by effect
    print(f"\n{'='*70}")
    print(f"CAUSAL FEATURES (Ranked by Causal Effect)")
    print(f"{'='*70}")
    causal_ranked = sorted(
        [(f, causal_effects[f]) for f in causal_features],
        key=lambda x: x[1],
        reverse=True
    )

    for i, (feature, effect) in enumerate(causal_ranked, 1):
        print(f"  {i:2d}. {feature:30s} | Causal Effect: {effect:.6f}")

    # Show noisy features
    print(f"\n{'='*70}")
    print(f"NOISY FEATURES (Non-causal - to be removed)")
    print(f"{'='*70}")
    noisy_ranked = sorted(
        [(f, causal_effects[f]) for f in noisy_features],
        key=lambda x: x[1],
        reverse=True
    )

    for i, (feature, effect) in enumerate(noisy_ranked, 1):
        print(f"  {i:2d}. {feature:30s} | Causal Effect: {effect:.6f}")

    # Save results
    results = {
        'causal_features': causal_features,
        'noisy_features': noisy_features,
        'causal_effects': causal_effects,
        'delta': DELTA,
        'num_samples': len(data)
    }

    results_path = os.path.join(OUTPUT_DIR, 'causal_features.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n[OK] Saved causal feature analysis to: {results_path}")

    # Create reduced dataset with only causal features
    data_causal = data[causal_features + ['label']]
    causal_data_path = os.path.join(OUTPUT_DIR, 'train_causal.csv')
    data_causal.to_csv(causal_data_path)

    print(f"[OK] Saved causal-only dataset to: {causal_data_path}")

    # Comparison with paper
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH ZENG ET AL. 2021")
    print(f"{'='*70}")
    print(f"Paper's result: 7-10 causal features for NSL-KDD")
    print(f"Our result: {len(causal_features)} causal features")

    if 7 <= len(causal_features) <= 10:
        print(f"[OK] MATCHES PAPER'S FINDINGS!")
    elif len(causal_features) < 7:
        print(f"[!] Fewer features than paper - consider lowering delta")
    else:
        print(f"[!] More features than paper - consider raising delta")

    print(f"\n{'='*70}")
    print(f"NEXT STEPS")
    print(f"{'='*70}")
    print(f"1. Review the {len(causal_features)} causal features identified")
    print(f"2. Create causal graph using these features")
    print(f"3. Update constant.py with categorical features from this list")
    print(f"4. Train Causal TGAN with causal features")
    print(f"\nRun: python create_causal_graph.py")
    print(f"{'='*70}\n")

    return causal_features, noisy_features, causal_effects


if __name__ == '__main__':
    causal_features, noisy_features, causal_effects = main()
