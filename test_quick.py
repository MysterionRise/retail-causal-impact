#!/usr/bin/env python
"""Quick test of core functionality."""
import sys
sys.path.insert(0, 'src')

from coupon_causal import data, utils

# Test data generation
config = utils.load_config('config/default.yaml')
utils.set_random_seed(42)

print("Testing synthetic data generation...")
df, gt = data.generate_synthetic_coupon_data(
    n_samples=100,
    treatment_rate=0.3,
    true_ate=15.0,
    random_state=42
)

print(f"✓ Data generation successful: {len(df)} rows")
print(f"✓ True ATE: ${gt['true_ate']:.2f}")
print(f"✓ Treatment rate: {df['treatment'].mean():.1%}")
print(f"✓ Mean outcome: ${df['outcome'].mean():.2f}")

print("\nAll tests passed!")
