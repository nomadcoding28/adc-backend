"""
tests/verify_cvar_math.py
===========================
Mathematical verification of CVaR computation.

This script verifies our CVaR implementation against analytical solutions
for known distributions, providing confidence that the math is correct
for the IEEE/NDSS paper submission.

Tests:
    1. Gaussian N(μ, σ²) → CVaR_α = μ − σ × φ(Φ⁻¹(α))/α
    2. Uniform U(a, b) → CVaR_α = a + (b − a) × α / 2
    3. Empirical vs. analytical comparison
    4. α-sensitivity analysis

Usage:
    python tests/verify_cvar_math.py
"""

from __future__ import annotations

import sys
import numpy as np
from scipy import stats
from pathlib import Path


def compute_empirical_cvar(returns: np.ndarray, alpha: float) -> float:
    """Compute empirical CVaR: mean of worst α-fraction of returns."""
    n_tail = max(1, int(alpha * len(returns)))
    return float(np.mean(np.sort(returns)[:n_tail]))


def compute_empirical_var(returns: np.ndarray, alpha: float) -> float:
    """Compute empirical VaR: α-quantile of the return distribution."""
    return float(np.percentile(returns, alpha * 100))


def analytical_cvar_gaussian(mu: float, sigma: float, alpha: float) -> float:
    """
    Analytical CVaR for Gaussian N(μ, σ²).

    CVaR_α = μ − σ × φ(Φ⁻¹(α)) / α

    where φ = standard normal PDF, Φ⁻¹ = standard normal quantile.
    """
    z_alpha = stats.norm.ppf(alpha)
    phi_z = stats.norm.pdf(z_alpha)
    return mu - sigma * phi_z / alpha


def analytical_cvar_uniform(a: float, b: float, alpha: float) -> float:
    """
    Analytical CVaR for Uniform U(a, b).

    CVaR_α = a + (b − a) × α / 2
    """
    return a + (b - a) * alpha / 2.0


def verify_gaussian_cvar(n_samples: int = 500_000) -> dict:
    """
    Verify CVaR for Gaussian distribution.

    Returns a dict of results for various (μ, σ, α) combinations.
    """
    rng = np.random.default_rng(42)
    results = []

    for mu, sigma in [(0.0, 1.0), (10.0, 5.0), (-5.0, 2.0)]:
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            samples = rng.normal(mu, sigma, n_samples).astype(np.float32)
            empirical = compute_empirical_cvar(samples, alpha)
            analytical = analytical_cvar_gaussian(mu, sigma, alpha)
            error = abs(empirical - analytical) / (abs(analytical) + 1e-8)

            # Use wider tolerance for extreme tails where sampling noise is larger
            tol = 0.15 if alpha <= 0.05 else 0.05
            results.append({
                "mu": mu, "sigma": sigma, "alpha": alpha,
                "empirical_cvar": round(empirical, 4),
                "analytical_cvar": round(analytical, 4),
                "relative_error": round(error, 6),
                "passed": error < tol,
            })

    return results


def verify_uniform_cvar(n_samples: int = 500_000) -> dict:
    """Verify CVaR for Uniform distribution."""
    rng = np.random.default_rng(42)
    results = []

    for a, b in [(0.0, 1.0), (-10.0, 10.0), (5.0, 100.0)]:
        for alpha in [0.01, 0.05, 0.10, 0.20]:
            samples = rng.uniform(a, b, n_samples).astype(np.float32)
            empirical = compute_empirical_cvar(samples, alpha)
            analytical = analytical_cvar_uniform(a, b, alpha)
            error = abs(empirical - analytical) / (abs(analytical) + 1e-8)

            tol = 0.15 if alpha <= 0.05 else 0.05
            results.append({
                "a": a, "b": b, "alpha": alpha,
                "empirical_cvar": round(empirical, 4),
                "analytical_cvar": round(analytical, 4),
                "relative_error": round(error, 6),
                "passed": error < tol,
            })

    return results


def verify_alpha_sensitivity() -> dict:
    """
    Generate the α-sensitivity table (Paper Table 2).

    Shows how CVaR changes across different α levels for a
    bimodal distribution (simulating good runs + catastrophic breaches).
    """
    rng = np.random.default_rng(42)

    # Bimodal: 90% good (μ=40, σ=10) + 10% catastrophic (μ=−30, σ=5)
    good = rng.normal(40.0, 10.0, 9000)
    bad  = rng.normal(-30.0, 5.0, 1000)
    returns = np.concatenate([good, bad]).astype(np.float32)

    alphas = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    rows = []

    for alpha in alphas:
        cvar = compute_empirical_cvar(returns, alpha)
        var  = compute_empirical_var(returns, alpha)
        rows.append({
            "alpha": alpha,
            "cvar": round(cvar, 4),
            "var": round(var, 4),
            "interpretation": _interpret_alpha(alpha),
        })

    return {
        "mean": round(float(np.mean(returns)), 4),
        "std": round(float(np.std(returns)), 4),
        "rows": rows,
    }


def _interpret_alpha(alpha: float) -> str:
    """Human-readable interpretation for a given alpha level."""
    labels = {
        0.01: "Extreme tail - worst 1% (ultra-conservative)",
        0.02: "Worst 2% - very conservative",
        0.05: "Default alpha - worst 5% (paper setting)",
        0.10: "Worst 10% - moderately conservative",
        0.20: "Worst 20% - balanced",
        0.50: "Worst 50% - median-aware",
        1.00: "Full mean - equivalent to standard PPO",
    }
    return labels.get(alpha, f"Worst {alpha*100:.0f}%")


def main() -> int:
    """Run all CVaR math verification tests."""
    print("=" * 72)
    print("CVaR Mathematical Verification - ACD Framework")
    print("=" * 72)

    # 1. Gaussian verification
    print("\n1. Gaussian CVaR Verification:")
    print("-" * 40)
    gaussian_results = verify_gaussian_cvar()
    for r in gaussian_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  N({r['mu']}, {r['sigma']}^2) alpha={r['alpha']:.2f}: "
              f"empirical={r['empirical_cvar']:+.4f}  "
              f"analytical={r['analytical_cvar']:+.4f}  "
              f"error={r['relative_error']:.4%}  {status}")

    # 2. Uniform verification
    print("\n2. Uniform CVaR Verification:")
    print("-" * 40)
    uniform_results = verify_uniform_cvar()
    for r in uniform_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  U({r['a']}, {r['b']}) alpha={r['alpha']:.2f}: "
              f"empirical={r['empirical_cvar']:+.4f}  "
              f"analytical={r['analytical_cvar']:+.4f}  "
              f"error={r['relative_error']:.4%}  {status}")

    # 3. alpha-sensitivity table
    print("\n3. Alpha-Sensitivity Table (Paper Table 2):")
    print("-" * 60)
    sensitivity = verify_alpha_sensitivity()
    print(f"  Distribution: Bimodal (mean={sensitivity['mean']}, std={sensitivity['std']})")
    print(f"  {'alpha':>6s}  {'CVaR':>10s}  {'VaR':>10s}  Interpretation")
    for row in sensitivity["rows"]:
        print(f"  {row['alpha']:>6.2f}  {row['cvar']:>+10.4f}  {row['var']:>+10.4f}  "
              f"{row['interpretation']}")

    # Summary
    all_passed = all(r["passed"] for r in gaussian_results + uniform_results)
    print(f"\n{'=' * 72}")
    print(f"RESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'=' * 72}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
