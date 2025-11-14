"""
Test NSL-KDD Setup
Verify that all components are ready for Causal TGAN training
"""

import os
import pickle
import pandas as pd
from helper.utils import load_data_graph, get_discrete_cols

def test_setup():
    print("="*70)
    print("TESTING NSL-KDD SETUP")
    print("="*70)

    data_name = 'nsl_kdd'

    # Test 1: Check data files exist
    print("\n[Test 1] Checking data files...")
    data_dir = './data/real_world/nsl_kdd'

    files_to_check = {
        'train_full.csv': 'Training data (all 42 features)',
        'test.csv': 'Test data (placeholder)',
        'graph.txt': 'Causal graph (10 CRFS features)',
        'causal_features.pkl': 'Causal analysis results'
    }

    all_exist = True
    for filename, description in files_to_check.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"  [OK] {filename:25s} - {description}")
        else:
            print(f"  [MISSING] {filename:25s} - {description}")
            all_exist = False

    if not all_exist:
        print("\n[ERROR] Some files are missing!")
        return False

    # Test 2: Load and validate data
    print("\n[Test 2] Loading training data...")
    try:
        data, col_names, discrete_cols, causal_graph = load_data_graph(data_name)
        print(f"  [OK] Data loaded: {len(data)} samples, {len(col_names)} columns")
        print(f"  [OK] Discrete columns: {discrete_cols}")
        print(f"  [OK] Causal graph loaded: {len(causal_graph)} nodes")
    except Exception as e:
        print(f"  [ERROR] Failed to load data: {e}")
        return False

    # Test 3: Validate causal graph structure
    print("\n[Test 3] Validating causal graph...")
    expected_features = [
        'service', 'dst_bytes', 'src_bytes', 'diff_srv_rate',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_serror_rate',
        'count', 'flag', 'label'
    ]

    graph_features = [node for node, _ in causal_graph]

    if set(graph_features) == set(expected_features):
        print(f"  [OK] All 11 features present in graph")
    else:
        missing = set(expected_features) - set(graph_features)
        extra = set(graph_features) - set(expected_features)
        if missing:
            print(f"  [ERROR] Missing features: {missing}")
        if extra:
            print(f"  [ERROR] Extra features: {extra}")
        return False

    # Test 4: Check data columns include causal graph features (partial knowledge mode)
    print("\n[Test 4] Validating data columns (partial knowledge mode)...")
    causal_features_in_data = set(expected_features) - {'label'}
    missing = causal_features_in_data - set(col_names)
    extra = set(col_names) - set(expected_features)

    if missing:
        print(f"  [ERROR] Data missing causal features: {missing}")
        return False
    else:
        print(f"  [OK] All 10 causal features present in data")

    if extra:
        print(f"  [OK] Data has {len(extra)} additional features (for Conditional GAN)")
        print(f"       Extra features: {sorted(list(extra))[:5]}... (showing first 5)")
        print(f"  [OK] PARTIAL KNOWLEDGE mode will be triggered")
        print(f"       - Causal Generator: 10 CRFS features")
        print(f"       - Conditional GAN: {len(extra)} remaining features")
    else:
        print(f"  [WARNING] No extra features - will run in full knowledge mode")

    # Test 5: Load causal features analysis
    print("\n[Test 5] Loading causal features analysis...")
    try:
        with open(os.path.join(data_dir, 'causal_features.pkl'), 'rb') as f:
            causal_results = pickle.load(f)

        causal_features = causal_results['causal_features']
        noisy_features = causal_results['noisy_features']

        print(f"  [OK] Causal features: {len(causal_features)}")
        print(f"       {causal_features}")
        print(f"  [OK] Noisy features removed: {len(noisy_features)}")
        print(f"       {noisy_features}")
    except Exception as e:
        print(f"  [ERROR] Failed to load causal analysis: {e}")
        return False

    # Test 6: Check label distribution
    print("\n[Test 6] Checking label distribution...")
    label_dist = data['label'].value_counts()
    print("  Label distribution:")
    for label, count in label_dist.items():
        print(f"    {label:10s}: {count:6d} ({count/len(data)*100:5.2f}%)")

    # Check for class imbalance
    minority_class = label_dist.min()
    majority_class = label_dist.max()
    imbalance_ratio = majority_class / minority_class

    if imbalance_ratio > 100:
        print(f"  [WARNING] Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        print(f"            Consider using balanced sampling or class weights")
    else:
        print(f"  [OK] Class imbalance ratio: {imbalance_ratio:.1f}:1")

    # Summary
    print("\n" + "="*70)
    print("SETUP VALIDATION COMPLETE")
    print("="*70)
    print("[OK] All tests passed!")
    print("\nReady to train Causal TGAN:")
    print("  python train.py --data_name nsl_kdd --epochs 400 --batch_size 500")
    print("="*70)

    return True


if __name__ == '__main__':
    success = test_setup()
    exit(0 if success else 1)
