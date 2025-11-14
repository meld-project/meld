"""
Statistical significance testing utilities for MELD experiments.

This module provides functions to perform statistical tests on experiment results,
helping determine whether observed differences are significant or due to chance.
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings

LOGGER = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Result of statistical test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""

class SignificanceTester:
    """Statistical significance testing for MELD experiments."""

    def __init__(self, alpha: float = 0.05, effect_size_threshold: float = 0.2):
        """
        Initialize significance tester.

        Args:
            alpha: Significance level (default 0.05)
            effect_size_threshold: Minimum effect size to consider meaningful
        """
        self.alpha = alpha
        self.effect_size_threshold = effect_size_threshold

    def compare_two_models(
        self,
        scores1: List[float],
        scores2: List[float],
        model1_name: str = "Model A",
        model2_name: str = "Model B",
        paired: bool = False
    ) -> TestResult:
        """
        Compare performance between two models.

        Args:
            scores1: Performance scores for model 1
            scores2: Performance scores for model 2
            model1_name: Name of first model
            model2_name: Name of second model
            paired: Whether samples are paired (same test sets)

        Returns:
            TestResult with statistical comparison
        """
        if len(scores1) != len(scores2) and paired:
            raise ValueError("Paired test requires equal number of samples")

        # Choose appropriate test
        if paired:
            # Paired samples t-test
            statistic, p_value = stats.ttest_rel(scores1, scores2)
            test_name = "Paired t-test"
        else:
            # Independent samples t-test
            statistic, p_value = stats.ttest_ind(scores1, scores2)
            test_name = "Independent t-test"

        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(scores1, scores2, paired)

        # Calculate confidence interval for difference
        ci = self._calculate_difference_ci(scores1, scores2, paired)

        # Interpret results
        is_significant = p_value < self.alpha
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        diff = mean1 - mean2

        if is_significant:
            if abs(effect_size) >= self.effect_size_threshold:
                interpretation = f"{model1_name} {'significantly outperforms' if diff > 0 else 'is significantly outperformed by'} {model2_name}"
            else:
                interpretation = f"Statistically significant but practically small difference between {model1_name} and {model2_name}"
        else:
            interpretation = f"No statistically significant difference between {model1_name} and {model2_name}"

        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation
        )

    def compare_multiple_models(
        self,
        model_scores: Dict[str, List[float]],
        metric_name: str = "performance"
    ) -> Dict[str, TestResult]:
        """
        Compare performance across multiple models using ANOVA and post-hoc tests.

        Args:
            model_scores: Dictionary mapping model names to score lists
            metric_name: Name of the metric being compared

        Returns:
            Dictionary of test results
        """
        if len(model_scores) < 2:
            raise ValueError("Need at least 2 models for comparison")

        results = {}

        # Perform ANOVA test
        scores_list = list(model_scores.values())
        model_names = list(model_scores.keys())

        f_stat, p_value = stats.f_oneway(*scores_list)

        results["anova"] = TestResult(
            test_name="One-way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            interpretation=f"{'Significant' if p_value < self.alpha else 'No significant'} differences among models"
        )

        # If ANOVA is significant, perform post-hoc pairwise comparisons
        if p_value < self.alpha:
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    name1, name2 = model_names[i], model_names[j]
                    pair_result = self.compare_two_models(
                        scores_list[i], scores_list[j], name1, name2
                    )
                    # Apply Bonferroni correction for multiple comparisons
                    corrected_alpha = self.alpha / (len(model_names) * (len(model_names) - 1) / 2)
                    pair_result.is_significant = pair_result.p_value < corrected_alpha
                    pair_result.interpretation += f" (Bonferroni corrected α={corrected_alpha:.4f})"
                    results[f"{name1}_vs_{name2}"] = pair_result

        return results

    def test_improvement_significance(
        self,
        baseline_scores: List[float],
        improved_scores: List[float],
        baseline_name: str = "Baseline",
        improved_name: str = "Improved"
    ) -> TestResult:
        """
        Test whether an improvement over baseline is significant.

        Args:
            baseline_scores: Baseline performance scores
            improved_scores: Improved performance scores
            baseline_name: Name of baseline method
            improved_name: Name of improved method

        Returns:
            TestResult focusing on improvement significance
        """
        result = self.compare_two_models(
            improved_scores, baseline_scores,
            improved_name, baseline_name, paired=True
        )

        # One-tailed test for improvement
        if len(improved_scores) == len(baseline_scores):
            improvements = [i - b for i, b in zip(improved_scores, baseline_scores)]
            mean_improvement = np.mean(improvements)

            # One-sample t-test against zero
            t_stat, p_value_one_tailed = stats.ttest_1samp(improvements, 0)
            p_value_one_tailed /= 2  # Convert to one-tailed

            if mean_improvement > 0:
                result.p_value = p_value_one_tailed
                result.is_significant = p_value_one_tailed < self.alpha
                result.test_name = "One-tailed paired t-test (improvement)"

                if result.is_significant:
                    result.interpretation = f"{improved_name} shows statistically significant improvement over {baseline_name}"
                else:
                    result.interpretation = f"{improved_name} does not show statistically significant improvement over {baseline_name}"

        return result

    def calculate_power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: Optional[float] = None,
        test_type: str = "two_sample"
    ) -> Dict[str, float]:
        """
        Calculate statistical power for given parameters.

        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size per group
            alpha: Significance level (uses instance default if None)
            test_type: Type of test ('two_sample', 'paired', 'one_sample')

        Returns:
            Dictionary with power analysis results
        """
        if alpha is None:
            alpha = self.alpha

        try:
            from statsmodels.stats.power import TTestIndPower, TTestPairedPower, TTestPower

            if test_type == "two_sample":
                power_analysis = TTestIndPower()
            elif test_type == "paired":
                power_analysis = TTestPairedPower()
            else:  # one_sample
                power_analysis = TTestPower()

            power = power_analysis.power(
                effect_size=effect_size,
                nobs1=sample_size,
                alpha=alpha,
                alternative="two-sided"
            )

            # Calculate required sample size for desired power
            required_n = power_analysis.solve_power(
                effect_size=effect_size,
                power=0.8,  # Desired power
                alpha=alpha,
                alternative="two-sided"
            )

            return {
                "power": power,
                "required_sample_size": required_n,
                "effect_size": effect_size,
                "sample_size": sample_size,
                "alpha": alpha
            }

        except ImportError:
            LOGGER.warning("statsmodels not available, power analysis skipped")
            return {
                "power": None,
                "required_sample_size": None,
                "effect_size": effect_size,
                "sample_size": sample_size,
                "alpha": alpha,
                "note": "Install statsmodels for power analysis"
            }

    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval.

        Args:
            scores: Performance scores
            confidence_level: Confidence level (0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        np.random.seed(42)  # For reproducibility
        n = len(scores)
        bootstrap_means = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        mean = np.mean(scores)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)

        return mean, lower_bound, upper_bound

    def _calculate_cohens_d(
        self,
        scores1: List[float],
        scores2: List[float],
        paired: bool = False
    ) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(scores1), np.mean(scores2)

        if paired:
            # For paired samples, use standard deviation of differences
            differences = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
            std_diff = np.std(differences, ddof=1)
            return (mean1 - mean2) / std_diff if std_diff > 0 else 0.0
        else:
            # For independent samples, use pooled standard deviation
            n1, n2 = len(scores1), len(scores2)
            var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    def _calculate_difference_ci(
        self,
        scores1: List[float],
        scores2: List[float],
        paired: bool = False,
        confidence: float = 0.95
    ) -> Optional[Tuple[float, float]]:
        """Calculate confidence interval for the difference."""
        try:
            alpha = 1 - confidence

            if paired:
                differences = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
                n = len(differences)
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                se = std_diff / np.sqrt(n)
                t_crit = stats.t.ppf(1 - alpha/2, n - 1)
                margin = t_crit * se
                return (mean_diff - margin, mean_diff + margin)
            else:
                n1, n2 = len(scores1), len(scores2)
                mean1, mean2 = np.mean(scores1), np.mean(scores2)
                var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)

                # Pooled standard error
                se = np.sqrt(var1/n1 + var2/n2)
                df = n1 + n2 - 2
                t_crit = stats.t.ppf(1 - alpha/2, df)
                margin = t_crit * se

                return (mean1 - mean2 - margin, mean1 - mean2 + margin)

        except Exception as e:
            LOGGER.warning(f"Failed to calculate confidence interval: {e}")
            return None

    def format_test_result(self, result: TestResult) -> str:
        """Format test result for display."""
        output = [
            f"Test: {result.test_name}",
            f"Statistic: {result.statistic:.4f}",
            f"P-value: {result.p_value:.4f}",
            f"Significant: {'Yes' if result.is_significant else 'No'}",
        ]

        if result.effect_size is not None:
            output.append(f"Effect size (Cohen's d): {result.effect_size:.4f}")

        if result.confidence_interval:
            lower, upper = result.confidence_interval
            output.append(f"95% CI: [{lower:.4f}, {upper:.4f}]")

        output.append(f"Interpretation: {result.interpretation}")

        return "\n".join(output)


# Convenience functions for common tests
def compare_auc_scores(
    auc1: List[float],
    auc2: List[float],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> TestResult:
    """Compare AUC scores between two models."""
    tester = SignificanceTester()
    return tester.compare_two_models(auc1, auc2, model1_name, model2_name)


def test_improvement_over_baseline(
    baseline_scores: List[float],
    new_scores: List[float],
    metric_name: str = "performance"
) -> TestResult:
    """Test if new method significantly improves over baseline."""
    tester = SignificanceTester()
    return tester.test_improvement_significance(
        baseline_scores, new_scores, "Baseline", "New Method"
    )


def multiple_model_comparison(
    model_scores: Dict[str, List[float]],
    metric_name: str = "performance"
) -> Dict[str, TestResult]:
    """Compare multiple models simultaneously."""
    tester = SignificanceTester()
    return tester.compare_multiple_models(model_scores, metric_name)


# Example usage
if __name__ == "__main__":
    # Example: Compare two models
    lec_scores = [0.82, 0.85, 0.83, 0.87, 0.84]
    baseline_scores = [0.78, 0.79, 0.77, 0.80, 0.76]

    tester = SignificanceTester()
    result = tester.compare_two_models(lec_scores, baseline_scores, "LEC", "Baseline")
    print(tester.format_test_result(result))

    # Example: Multiple model comparison
    model_scores = {
        "LEC": [0.82, 0.85, 0.83, 0.87, 0.84],
        "BERT": [0.81, 0.83, 0.82, 0.84, 0.83],
        "TF-IDF": [0.78, 0.79, 0.77, 0.80, 0.76]
    }

    results = multiple_model_comparison(model_scores)
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        print(tester.format_test_result(result))