# Synthetic Data Quality Evaluation

## Overview

This script (`evaluate_synthetic_data_comprehensive.py`) implements **all metrics** from the paper:
**"Synthetic Network Traffic Data Generation: A Comparative Study"** by Ammara et al. (2025)

## Metrics Implemented

### 1. Statistical Similarity (Fidelity)
- **Data Structure (DS)**: Validates that binary/boolean columns maintain expected values (0 or 1)
- **Correlation (Corr)**: Compares correlation matrices between real and synthetic data
- **Probability Distribution (PD)**: Uses Kolmogorov-Smirnov test to compare distributions

### 2. Machine Learning Utility
- **TRTR (Train on Real, Test on Real)**: Baseline performance
- **TSTR (Train on Synthetic, Test on Real)**: Synthetic data utility
- **Utility Ratio**: TSTR/TRTR performance comparison

### 3. Class Balance
- **CB**: Evaluates distribution across classes (Normal vs Attack)

### 4. Visualizations
- Correlation heatmaps (Real, Synthetic, Difference)
- Probability distributions for top features
- Class balance bar charts

## Usage

### Basic Usage

```bash
python evaluate_synthetic_data_comprehensive.py
```

### Custom Paths

Edit the paths in the `main()` function:

```python
REAL_TRAIN_PATH = "data/real_world/nsl_kdd/train_full.csv"
SYNTHETIC_PATH = "Testing/CausalTGAN_runs_new_3_nsl_kdd2025.11.13--13-57-06/generated_samples.csv"
REAL_TEST_PATH = "data/real_world/nsl_kdd/test_full.csv"
OUTPUT_DIR = "evaluation_results"
```

### Programmatic Usage

```python
from evaluate_synthetic_data_comprehensive import SyntheticDataEvaluator

evaluator = SyntheticDataEvaluator(
    real_train_path="path/to/real_train.csv",
    synthetic_path="path/to/synthetic.csv",
    real_test_path="path/to/real_test.csv",
    output_dir="my_evaluation_results"
)

results = evaluator.run_full_evaluation()
```

## Output

The script generates:

1. **Console Output**: Real-time progress and results
2. **Text Report**: `evaluation_results/evaluation_report.txt`
3. **Visualizations**:
   - `correlation_heatmaps.png`
   - `probability_distributions.png`
   - `class_balance.png`

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Interpretation Guide

### Data Structure (DS)
- **YES**: All binary columns have valid values (0 or 1)
- **NO**: Some binary columns have invalid values

### Correlation (Corr)
- **YES**: Mean correlation difference < 0.1
- **NO**: Mean correlation difference e 0.1

### Probability Distribution (PD)
- **0% diff**: All features have same distributions
- **>0% diff**: Percentage of features with significantly different distributions

### Class Balance (CB)
- **0% diff**: Perfect balance
- **>0% diff**: Imbalance between classes

### ML Utility Ratio
- **e95%**: EXCELLENT - Synthetic data captures >95% of real data utility
- **90-95%**: VERY GOOD - Captures >90% of utility
- **80-90%**: GOOD - Captures >80% of utility
- **70-80%**: FAIR - Captures >70% of patterns
- **<70%**: NEEDS IMPROVEMENT

## Comparison with Paper Results

The paper evaluated 12 methods on NSL-KDD and CIC-IDS2017:

| Method | DS | Corr | PD Diff | CB Diff | Accuracy (TSTR) |
|--------|-----|------|---------|---------|-----------------|
| CTGAN | YES | YES | 7.7% | 6.7% | 0.9667 |
| CopulaGAN | YES | YES | 7.7% | 5.4% | 0.9761 |
| TVAE | YES | YES | 23.1% | 26.7% | 0.9807 |
| SMOTE | NO | YES | 0% | 0% | 0.9999 |

Your results will be compared against these benchmarks.

## Example Output

```
================================================================================
SYNTHETIC DATA QUALITY EVALUATION - SUMMARY REPORT
================================================================================

STATISTICAL SIMILARITY (FIDELITY)
--------------------------------------------------------------------------------
Data Structure (DS)            : YES (DS Violations: 0)
Correlation (Corr)             : YES (Corr Mean Diff: 0.0450)
Probability Distribution (PD)  : 15.2% diff (PD Different Count: 6)

CLASS BALANCE
--------------------------------------------------------------------------------
Class Balance Difference       : 2.34% diff

MACHINE LEARNING UTILITY
--------------------------------------------------------------------------------
TRTR Accuracy                  : 0.9812
TSTR Accuracy                  : 0.9654
Utility Ratio                  : 98.39%

OVERALL ASSESSMENT
--------------------------------------------------------------------------------
Overall Quality Score: 91.5/100
Grade: EXCELLENT
```

## Reference

```bibtex
@article{ammara2025synthetic,
  title={Synthetic Network Traffic Data Generation: A Comparative Study},
  author={Ammara, Dure Adan and Ding, Jianguo and Tutschku, Kurt},
  journal={arXiv preprint arXiv:2410.16326},
  year={2025}
}
```

## Notes

- The script uses Random Forest classifier with 100 trees (matching the paper)
- Correlation threshold is set to 0.1 (10% mean absolute difference)
- KS-test significance level is 0.05
- Categorical variables (protocol_type, service, flag) are encoded using LabelEncoder
