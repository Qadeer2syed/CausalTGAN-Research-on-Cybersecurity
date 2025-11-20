"""
Comprehensive Synthetic Data Quality Evaluation Script
Based on: "Synthetic Network Traffic Data Generation: A Comparative Study"

Metrics Implemented:
1. Statistical Similarity (Fidelity)
   - Data Structure (DS): Binary/boolean value validation
   - Correlation (Corr): Correlation matrix comparison
   - Probability Distribution (PD): KDE and KS-test comparison

2. ML Utility (Performance/Accuracy)
   - TRTR: Train on Real, Test on Real
   - TSTR: Train on Synthetic, Test on Real

3. Class Balance (CB)
   - Distribution across classes

4. Visualizations
   - Correlation heatmaps
   - Probability distributions
   - Class balance charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

class SyntheticDataEvaluator:
    """
    Comprehensive evaluator for synthetic tabular data quality
    """

    def __init__(self, real_train_path, synthetic_path, real_test_path=None, output_dir='evaluation_results'):
        """
        Initialize evaluator

        Args:
            real_train_path: Path to real training data
            synthetic_path: Path to synthetic data
            real_test_path: Path to real test data (optional)
            output_dir: Directory to save results
        """
        print("="*80)
        print("SYNTHETIC DATA QUALITY EVALUATOR")
        print("="*80)
        print("\n[1/4] Loading datasets...")

        # Load data
        self.real_train = pd.read_csv(real_train_path)
        self.synthetic = pd.read_csv(synthetic_path)

        if real_test_path:
            self.real_test = pd.read_csv(real_test_path)
        else:
            self.real_test = None

        # Remove index column if exists
        if self.real_train.columns[0] == 'Unnamed: 0' or self.real_train.iloc[:, 0].name == '':
            self.real_train = self.real_train.iloc[:, 1:]
        if self.real_test is not None and (self.real_test.columns[0] == 'Unnamed: 0' or self.real_test.iloc[:, 0].name == ''):
            self.real_test = self.real_test.iloc[:, 1:]

        print(f"   Real training data: {self.real_train.shape}")
        print(f"   Synthetic data: {self.synthetic.shape}")
        if self.real_test is not None:
            print(f"   Real test data: {self.real_test.shape}")

        # Setup
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Identify target column (last column)
        self.target_col = 'label'

        # Identify categorical columns
        self.categorical_cols = ['protocol_type', 'service', 'flag']

        # Results storage
        self.results = {}

    def evaluate_data_structure(self):
        """
        Metric 1a: Data Structure (DS)
        Checks if binary/boolean columns maintain expected values
        """
        print("\n[2/4] Evaluating Data Structure...")

        violations = []

        # Check binary columns
        binary_cols = []
        for col in self.synthetic.columns:
            if col == self.target_col or col in self.categorical_cols:
                continue

            real_unique = set(self.real_train[col].dropna().unique())

            # If real data has binary values
            if real_unique.issubset({0, 1}) or real_unique.issubset({0.0, 1.0}):
                binary_cols.append(col)
                synth_unique = set(self.synthetic[col].dropna().unique())

                # Check if synthetic maintains binary constraint
                synth_non_binary = synth_unique - {0, 1, 0.0, 1.0}
                if len(synth_non_binary) > 0:
                    violations.append({
                        'column': col,
                        'invalid_values': synth_non_binary
                    })

        ds_result = "YES" if len(violations) == 0 else "NO"
        self.results['Data Structure (DS)'] = ds_result
        self.results['DS Violations'] = len(violations)

        print(f"  Binary columns checked: {len(binary_cols)}")
        print(f"  Violations found: {len(violations)}")
        print(f"  Result: {ds_result}")

        if violations and len(violations) <= 5:
            print("\n  Violation details:")
            for v in violations[:5]:
                print(f"    • {v['column']}: has values {v['invalid_values']}")

        return ds_result

    def evaluate_correlation(self):
        """
        Metric 1b: Correlation (Corr)
        Compares correlation matrices between real and synthetic data
        """
        print("\n[3/4] Evaluating Correlation...")

        # Get numerical columns only
        numeric_cols = self.real_train.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        # Compute correlation matrices
        corr_real = self.real_train[numeric_cols].corr()
        corr_synth = self.synthetic[numeric_cols].corr()

        # Compute absolute difference
        corr_diff = np.abs(corr_real - corr_synth)

        # Mean absolute difference (excluding diagonal)
        mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
        mean_diff = corr_diff.where(mask).mean().mean()

        # Determine result (threshold: 0.1)
        corr_result = "YES" if mean_diff < 0.1 else "NO"

        self.results['Correlation (Corr)'] = corr_result
        self.results['Corr Mean Diff'] = f"{mean_diff:.4f}"

        print(f"  Mean absolute correlation difference: {mean_diff:.4f}")
        print(f"  Result: {corr_result}")

        # Save correlation heatmaps
        self._plot_correlation_heatmaps(corr_real, corr_synth, corr_diff)

        return corr_result

    def _plot_correlation_heatmaps(self, corr_real, corr_synth, corr_diff):
        """Plot and save correlation heatmaps"""
        # Limit to top 20 features for readability
        if len(corr_real) > 20:
            # Select features with highest variance in real data
            variances = self.real_train[corr_real.columns].var().sort_values(ascending=False)
            top_features = variances.head(20).index.tolist()
            corr_real = corr_real.loc[top_features, top_features]
            corr_synth = corr_synth.loc[top_features, top_features]
            corr_diff = corr_diff.loc[top_features, top_features]

        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        # Real correlation
        sns.heatmap(corr_real, ax=axes[0], cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'},
                    xticklabels=True, yticklabels=True)
        axes[0].set_title('Real Data Correlation Matrix', fontsize=14, fontweight='bold')

        # Synthetic correlation
        sns.heatmap(corr_synth, ax=axes[1], cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'},
                    xticklabels=True, yticklabels=True)
        axes[1].set_title('Synthetic Data Correlation Matrix', fontsize=14, fontweight='bold')

        # Difference
        sns.heatmap(corr_diff, ax=axes[2], cmap='Reds',
                    vmin=0, vmax=1, square=True, cbar_kws={'label': 'Abs Difference'},
                    xticklabels=True, yticklabels=True)
        axes[2].set_title('Absolute Correlation Difference', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        print(f"   Correlation heatmaps saved")
        plt.close()

    def evaluate_probability_distribution(self):
        """
        Metric 1c: Probability Distribution (PD)
        Compares distributions using KS test
        """
        print("\n[4/4] Evaluating Probability Distributions...")

        # Get numerical columns
        numeric_cols = self.real_train.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        different_distributions = []
        ks_results = {}

        for col in numeric_cols:
            try:
                real_vals = self.real_train[col].dropna()
                synth_vals = self.synthetic[col].dropna()

                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(real_vals, synth_vals)
                ks_results[col] = {'ks_stat': ks_stat, 'p_value': p_value}

                # p < 0.05 means significantly different
                if p_value < 0.05:
                    different_distributions.append(col)

            except Exception as e:
                print(f"  Warning: Could not test {col}: {e}")

        # Calculate percentage
        pd_diff_pct = (len(different_distributions) / len(numeric_cols)) * 100

        self.results['Probability Distribution (PD)'] = f"{pd_diff_pct:.1f}% diff"
        self.results['PD Different Count'] = len(different_distributions)

        print(f"  Features with different distributions: {len(different_distributions)}/{len(numeric_cols)}")
        print(f"  Percentage: {pd_diff_pct:.1f}%")

        # Plot distributions for top different features
        self._plot_probability_distributions(ks_results, different_distributions)

        return pd_diff_pct

    def _plot_probability_distributions(self, ks_results, different_distributions, n_plots=16):
        """Plot probability distributions"""
        # Select features to plot (prioritize different ones)
        if len(different_distributions) > 0:
            features_to_plot = different_distributions[:n_plots]
        else:
            # Plot random features
            features_to_plot = list(ks_results.keys())[:n_plots]

        n_features = len(features_to_plot)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, col in enumerate(features_to_plot):
            ax = axes[idx]

            real_vals = self.real_train[col].dropna()
            synth_vals = self.synthetic[col].dropna()

            try:
                # Check if binary/categorical
                if len(real_vals.unique()) <= 10:
                    # Bar plot for discrete
                    real_counts = real_vals.value_counts(normalize=True).sort_index()
                    synth_counts = synth_vals.value_counts(normalize=True).sort_index()

                    x = np.arange(len(real_counts))
                    width = 0.35
                    ax.bar(x - width/2, real_counts.values, width, label='Real', alpha=0.7, color='blue')
                    ax.bar(x + width/2, synth_counts.values, width, label='Synthetic', alpha=0.7, color='red')
                    ax.set_xticks(x)
                    ax.set_xticklabels(real_counts.index)
                else:
                    # KDE for continuous
                    real_vals.plot(kind='density', ax=ax, label='Real', color='blue', alpha=0.7)
                    synth_vals.plot(kind='density', ax=ax, label='Synthetic', color='red', alpha=0.7)

                ks_stat = ks_results[col]['ks_stat']
                p_val = ks_results[col]['p_value']

                ax.set_title(f'{col}\nKS={ks_stat:.3f}, p={p_val:.4f}', fontsize=9)
                ax.set_xlabel('')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{col}', ha='center', va='center')

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/probability_distributions.png', dpi=300, bbox_inches='tight')
        print(f"   Probability distribution plots saved")
        plt.close()

    def evaluate_class_balance(self):
        """
        Metric 3: Class Balance (CB)
        Evaluates distribution across classes
        """
        print("\n[5/7] Evaluating Class Balance...")

        # Get class distributions
        real_dist = self.real_train[self.target_col].value_counts(normalize=True) * 100
        synth_dist = self.synthetic[self.target_col].value_counts(normalize=True) * 100

        print("\n  Real data class distribution:")
        for cls, pct in real_dist.items():
            print(f"    {cls:15s}: {pct:6.2f}%")

        print("\n  Synthetic data class distribution:")
        for cls, pct in synth_dist.items():
            print(f"    {cls:15s}: {pct:6.2f}%")

        # Calculate balance metric
        all_classes = list(set(real_dist.index) | set(synth_dist.index))

        if len(all_classes) == 2:
            # Binary: difference between two classes
            class_diff = abs(synth_dist.iloc[0] - synth_dist.iloc[1])
            cb_result = f"{class_diff:.2f}% diff"
        else:
            # Multi-class: average absolute difference
            diffs = []
            for cls in all_classes:
                real_pct = real_dist.get(cls, 0)
                synth_pct = synth_dist.get(cls, 0)
                diffs.append(abs(real_pct - synth_pct))
            cb_result = f"{np.mean(diffs):.2f}% avg diff"

        self.results['Class Balance (CB)'] = cb_result

        print(f"\n  Class balance: {cb_result}")

        # Plot class balance
        self._plot_class_balance(real_dist, synth_dist)

        return cb_result

    def _plot_class_balance(self, real_dist, synth_dist):
        """Plot class balance comparison"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        x = np.arange(len(real_dist))
        width = 0.35

        ax.bar(x - width/2, real_dist.values, width, label='Real', alpha=0.7, color='blue')
        ax.bar(x + width/2, synth_dist.values, width, label='Synthetic', alpha=0.7, color='red')

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(real_dist.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/class_balance.png', dpi=300, bbox_inches='tight')
        print(f"   Class balance plot saved")
        plt.close()

    def evaluate_ml_utility(self):
        """
        Metric 2: ML Utility
        Compares TRTR vs TSTR performance
        """
        print("\n[6/7] Evaluating Machine Learning Utility...")

        # Prepare data
        if self.real_test is not None:
            X_test = self.real_test.drop(columns=[self.target_col])
            y_test = self.real_test[self.target_col]
        else:
            # Split real data
            X_test, _, y_test, _ = train_test_split(
                self.real_train.drop(columns=[self.target_col]),
                self.real_train[self.target_col],
                test_size=0.3, random_state=42
            )

        # Encode categorical variables
        X_test_encoded, real_train_encoded, synthetic_encoded = self._encode_categorical(X_test)

        # TRTR: Train on Real, Test on Real
        print("\n  [TRTR] Training on real data...")
        X_train_real = real_train_encoded.drop(columns=[self.target_col])
        y_train_real = real_train_encoded[self.target_col]

        model_trtr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        model_trtr.fit(X_train_real, y_train_real)
        y_pred_trtr = model_trtr.predict(X_test_encoded)

        trtr_acc = accuracy_score(y_test, y_pred_trtr)
        trtr_f1 = f1_score(y_test, y_pred_trtr, average='weighted')

        print(f"    Accuracy: {trtr_acc:.4f}")
        print(f"    F1 Score: {trtr_f1:.4f}")

        # TSTR: Train on Synthetic, Test on Real
        print("\n  [TSTR] Training on synthetic data...")
        X_train_synth = synthetic_encoded.drop(columns=[self.target_col])
        y_train_synth = synthetic_encoded[self.target_col]

        model_tstr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        model_tstr.fit(X_train_synth, y_train_synth)
        y_pred_tstr = model_tstr.predict(X_test_encoded)

        tstr_acc = accuracy_score(y_test, y_pred_tstr)
        tstr_f1 = f1_score(y_test, y_pred_tstr, average='weighted')

        print(f"    Accuracy: {tstr_acc:.4f}")
        print(f"    F1 Score: {tstr_f1:.4f}")

        # Calculate utility ratio
        utility_ratio = tstr_acc / trtr_acc * 100

        self.results['TRTR Accuracy'] = f"{trtr_acc:.4f}"
        self.results['TSTR Accuracy'] = f"{tstr_acc:.4f}"
        self.results['Utility Ratio'] = f"{utility_ratio:.2f}%"

        print(f"\n  Utility Ratio (TSTR/TRTR): {utility_ratio:.2f}%")

        return trtr_acc, tstr_acc

    def _encode_categorical(self, X_test):
        """Encode categorical variables consistently"""
        real_train_encoded = self.real_train.copy()
        synthetic_encoded = self.synthetic.copy()
        X_test_encoded = X_test.copy()

        for col in self.categorical_cols:
            if col in real_train_encoded.columns:
                le = LabelEncoder()

                # Fit on combined data
                combined = pd.concat([
                    self.real_train[col],
                    self.synthetic[col],
                    X_test[col]
                ])
                le.fit(combined)

                # Transform
                real_train_encoded[col] = le.transform(self.real_train[col])
                synthetic_encoded[col] = le.transform(self.synthetic[col])
                X_test_encoded[col] = le.transform(X_test[col])

        return X_test_encoded, real_train_encoded, synthetic_encoded

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n[7/7] Generating Summary Report...")

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SYNTHETIC DATA QUALITY EVALUATION - SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append(f"Real Training Data:    {self.real_train.shape[0]:,} samples, {self.real_train.shape[1]} features")
        report_lines.append(f"Synthetic Data:        {self.synthetic.shape[0]:,} samples, {self.synthetic.shape[1]} features")
        if self.real_test is not None:
            report_lines.append(f"Real Test Data:        {self.real_test.shape[0]:,} samples")
        report_lines.append("")

        report_lines.append("-"*80)
        report_lines.append("STATISTICAL SIMILARITY (FIDELITY)")
        report_lines.append("-"*80)

        metrics = [
            ('Data Structure (DS)', 'DS Violations'),
            ('Correlation (Corr)', 'Corr Mean Diff'),
            ('Probability Distribution (PD)', 'PD Different Count')
        ]

        for metric_name, detail_key in metrics:
            if metric_name in self.results:
                line = f"{metric_name:35s}: {self.results[metric_name]}"
                if detail_key in self.results:
                    line += f" ({detail_key}: {self.results[detail_key]})"
                report_lines.append(line)

        report_lines.append("")
        report_lines.append("-"*80)
        report_lines.append("CLASS BALANCE")
        report_lines.append("-"*80)
        if 'Class Balance (CB)' in self.results:
            report_lines.append(f"{'Class Balance Difference':35s}: {self.results['Class Balance (CB)']}")

        report_lines.append("")
        report_lines.append("-"*80)
        report_lines.append("MACHINE LEARNING UTILITY")
        report_lines.append("-"*80)

        for key in ['TRTR Accuracy', 'TSTR Accuracy', 'Utility Ratio']:
            if key in self.results:
                report_lines.append(f"{key:35s}: {self.results[key]}")

        report_lines.append("")
        report_lines.append("-"*80)
        report_lines.append("OVERALL ASSESSMENT")
        report_lines.append("-"*80)

        # Generate overall grade
        scores = []
        if self.results.get('Data Structure (DS)') == 'YES':
            scores.append(100)
        else:
            scores.append(0)

        if self.results.get('Correlation (Corr)') == 'YES':
            scores.append(100)
        else:
            scores.append(50)

        # PD score (inverse of difference percentage)
        pd_val = self.results.get('Probability Distribution (PD)', '0% diff')
        pd_pct = float(pd_val.split('%')[0].strip())
        scores.append(max(0, 100 - pd_pct))

        # Utility score
        if 'Utility Ratio' in self.results:
            utility_val = float(self.results['Utility Ratio'].strip('%'))
            scores.append(min(100, utility_val))

        overall_score = np.mean(scores)

        if overall_score >= 90:
            grade = "EXCELLENT"
        elif overall_score >= 80:
            grade = "VERY GOOD"
        elif overall_score >= 70:
            grade = "GOOD"
        elif overall_score >= 60:
            grade = "FAIR"
        else:
            grade = "NEEDS IMPROVEMENT"

        report_lines.append(f"Overall Quality Score: {overall_score:.1f}/100")
        report_lines.append(f"Grade: {grade}")
        report_lines.append("")
        report_lines.append("="*80)

        # Print to console
        print("")
        for line in report_lines:
            print(line)

        # Save to file
        report_path = f"{self.output_dir}/evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"\n Full report saved to: {report_path}")
        print(f" All visualizations saved to: {self.output_dir}/")

        return self.results

    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EVALUATION")
        print("="*80)

        # Run all metrics
        self.evaluate_data_structure()
        self.evaluate_correlation()
        self.evaluate_probability_distribution()
        self.evaluate_class_balance()
        self.evaluate_ml_utility()
        self.generate_summary_report()

        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)

        return self.results


def main():
    """Main execution function"""

    # Configuration
    REAL_TRAIN_PATH = "data/real_world/nsl_kdd/train_full.csv"
    SYNTHETIC_PATH = "Testing/CausalTGAN_runs_new_3_nsl_kdd2025.11.13--13-57-06/generated_samples.csv"
    REAL_TEST_PATH = "data/real_world/nsl_kdd/test_full.csv"
    OUTPUT_DIR = "evaluation_results"

    # Create evaluator
    evaluator = SyntheticDataEvaluator(
        real_train_path=REAL_TRAIN_PATH,
        synthetic_path=SYNTHETIC_PATH,
        real_test_path=REAL_TEST_PATH,
        output_dir=OUTPUT_DIR
    )

    # Run evaluation
    results = evaluator.run_full_evaluation()

    return results


if __name__ == "__main__":
    results = main()
