# Coupon Causal Impact Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/MysterionRise/retail-causal-impact/workflows/CI/badge.svg)](https://github.com/MysterionRise/retail-causal-impact/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/MysterionRise/retail-causal-impact/branch/main/graph/badge.svg)](https://codecov.io/gh/MysterionRise/retail-causal-impact)

Production-quality causal impact analysis for retail coupon campaigns. Estimate average and heterogeneous treatment effects (ATE, CATE), evaluate targeting policies, and optimize coupon allocation using state-of-the-art causal inference methods.

## 🎯 Features

- **Causal Estimation**
  - Average Treatment Effect (ATE): IPW, AIPW (doubly-robust)
  - Conditional ATE (CATE): DRLearner, Causal Forest (CausalForestDML), X-Learner
  - Propensity score modeling with diagnostics
  - Covariate balance assessment (Love plots, SMD)

- **Policy Learning & Uplift**
  - Qini curves and Area Under Uplift Curve (AUUC)
  - Budget-constrained targeting recommendations
  - Segment-level uplift analysis
  - ROI and net benefit evaluation

- **Robustness & Sensitivity**
  - DoWhy causal graph and refutation tests
  - Placebo tests, leave-one-out analysis
  - Bootstrap confidence intervals
  - Synthetic data with known ground truth

- **Production-Ready**
  - Clean, modular Python package
  - CLI for end-to-end pipelines
  - Interactive Streamlit dashboard
  - Comprehensive test suite
  - Reproducible with fixed seeds and pinned dependencies

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Streamlit Dashboard](#streamlit-dashboard)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Assumptions & Limitations](#assumptions--limitations)
- [Results](#results)
- [References](#references)

## 🚀 Installation

### Prerequisites

- Python 3.11, 3.12 or 3.13
- pip

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd retail-causal-impact

# Install dependencies and setup
make setup

# Or manually:
pip install -e ".[dev]"
mkdir -p data/{raw,interim,processed} reports/{figures,tables} models
```

### Dependencies

Core libraries:
- **Causal inference**: `econml`, `dowhy`
- **ML/Stats**: `scikit-learn`, `statsmodels`, `numpy`, `pandas`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Dashboard**: `streamlit`
- **Testing**: `pytest`, `pytest-cov`

See `pyproject.toml` for full dependency list with pinned versions.

## ⚡ Quick Start

### End-to-End Pipeline

Run the complete pipeline on synthetic data:

```bash
# Run the pipeline: data → estimation → reports (fast demo profile, ~1.5 min on 4 cores)
make all

# Or step-by-step:
make data        # Generate synthetic data
make train       # Run causal estimation
make report      # Generate plots and tables

# Heavier profile: 500 bootstrap draws, bigger forests, 5-fold cross-fitting (~15-25 min)
make full
```

Two config profiles are provided: `config/default.yaml` (fast, used by `make all`) and
`config/full.yaml` (thorough). Both run every estimator; they differ only in bootstrap
draws, model sizes and cross-fitting folds.

### Launch Dashboard

```bash
make app
# Opens at http://localhost:8501
```

### Run Tests

```bash
# Run all tests
make test

# Or with pytest directly
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/coupon_causal --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_synth_recovery.py -v
```

**Test Suite:**
- `test_synth_recovery.py` - Validates ATE/CATE recovery on synthetic data with known ground truth
- `test_balance.py` - Tests covariate balance improvement after propensity weighting
- `test_policy.py` - Tests policy learning and uplift evaluation (Qini, AUUC)

17 tests, all passing on Python 3.11-3.13 with NumPy 2.4+.

## 📖 Usage

### Command Line

#### Generate and Analyze Synthetic Data

```bash
# Run pipeline on synthetic data (default)
python -m scripts.run_pipeline --config config/default.yaml --data_source synth

# Run specific stages
python -m scripts.run_pipeline --data_source synth --stage data
python -m scripts.run_pipeline --data_source synth --stage estimation
python -m scripts.run_pipeline --data_source synth --stage report
```

#### Analyze Public Dataset

```bash
# Place your dataset in data/raw/coupons.csv
# Required columns: treatment (0/1), outcome (continuous), features

python -m scripts.run_pipeline \
  --config config/default.yaml \
  --data_source public \
  --input data/raw/coupons.csv
```

#### Score and Deploy Targeting Policy

```bash
# Generate targeting recommendations
python -m scripts.score_policy \
  --scores reports/tables/cate_scores.parquet \
  --budget 0.2 \
  --cost_per_coupon 1.0 \
  --output reports/tables/targeting_recommendations.csv
```

### Jupyter Notebooks

Three step-by-step notebooks are provided:

1. **`notebooks/00_eda.ipynb`** - Exploratory Data Analysis
   - Treatment and outcome distributions
   - Covariate balance checks
   - Segment analysis

2. **`notebooks/10_causal_estimation.ipynb`** - Causal Estimation
   - Propensity score modeling
   - ATE estimation (naive, IPW, AIPW)
   - CATE estimation (DRLearner, Causal Forest, X-Learner)
   - Balance diagnostics

3. **`notebooks/20_policy_uplift.ipynb`** - Policy & Uplift
   - Qini curves and AUUC
   - Budget-constrained policy evaluation
   - Segment uplift analysis
   - Targeting recommendations

Run notebooks:

```bash
jupyter notebook notebooks/
```

### Streamlit Dashboard

Interactive dashboard for exploring results:

```bash
streamlit run app/streamlit_app.py
```

Features:
- Data source selection (synthetic / public)
- ATE estimates with confidence intervals
- CATE distribution and Qini curves
- Segment analysis
- Downloadable targeting lists

## 📁 Project Structure

```
retail-causal-impact/
├── config/
│   ├── default.yaml          # Fast demo profile (seeds, hyperparameters, paths)
│   └── full.yaml             # Thorough profile (500 bootstrap draws, larger forests)
├── data/
│   ├── raw/                  # Place public datasets here
│   ├── interim/
│   └── processed/            # Generated/cleaned data
├── src/coupon_causal/        # Main Python package
│   ├── data.py               # Data loading & synthetic generation
│   ├── features.py           # Feature engineering
│   ├── propensity.py         # Propensity score models
│   ├── ate.py                # ATE estimation (IPW, AIPW)
│   ├── cate.py               # CATE estimation (DRLearner, Causal Forest, X-Learner)
│   ├── policy.py             # Policy learning, Qini, AUUC
│   ├── balance.py            # Balance diagnostics (SMD, Love plots)
│   ├── refute.py             # Refutation & sensitivity (DoWhy)
│   ├── viz.py                # Visualization utilities
│   └── utils.py              # Utilities (logging, config, seeds)
├── scripts/
│   ├── run_pipeline.py       # CLI orchestrator
│   └── score_policy.py       # Policy scoring CLI
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 10_causal_estimation.ipynb
│   └── 20_policy_uplift.ipynb
├── app/
│   └── streamlit_app.py      # Interactive dashboard
├── tests/
│   ├── test_synth_recovery.py  # ATE/CATE recovery on synthetic data
│   ├── test_balance.py         # Balance improvement tests
│   └── test_policy.py          # Policy uplift tests
├── reports/
│   ├── figures/              # Generated plots
│   └── tables/               # Generated CSVs
├── models/                   # Saved models and results
├── Makefile                  # Build automation
├── pyproject.toml            # Dependencies & package config
├── LICENSE                   # MIT
└── README.md                 # This file
```

## 🧪 Methodology

### Data Generation (Synthetic Mode)

The synthetic data generator creates realistic confounded coupon campaign data:

1. **Customer Features**: Loyalty, recency, frequency, monetary value, price sensitivity, segments
2. **Confounded Treatment Assignment**: A propensity score p(X) rises with loyalty, recency,
   frequency and the high-value segment (companies target loyal customers). Treatment is
   *drawn* as T ~ Bernoulli(p(X)), so every customer has a non-zero chance of either arm
   and the positivity assumption holds. The logit intercept is calibrated to the target
   treatment rate.
3. **Heterogeneous Treatment Effects**: Effect varies by price sensitivity, loyalty and segment
   (new and low-value customers respond most; loyal customers least)
4. **Known Ground Truth**: True propensity, true CATE and true ATE are stored for validation

### Causal Estimation Pipeline

1. **Feature Engineering**
   - One-hot encoding for categorical variables
   - Standardization for continuous variables
   - Leakage prevention (drop post-treatment variables)

2. **Propensity Score Modeling**
   - Logistic regression and gradient boosting are both fit (out-of-fold predictions)
   - The model used for weighting and ATE is set by `propensity.ate_model`
     (default `logistic`, which is also what the bootstrap refits on every draw)
   - Calibration diagnostics, overlap assessment
   - Clipping to [0.01, 0.99] for stability

3. **ATE Estimation**
   - **Naive**: Simple difference in means (biased baseline)
   - **IPW**: Inverse propensity weighting with stabilization
   - **AIPW (Doubly-Robust)**: Combines outcome models + IPW
     - Linear and Random Forest outcome models
   - Percentile bootstrap 95% confidence intervals, refitting the propensity model on
     each draw (100 draws in the default profile, 500 in `config/full.yaml`)

4. **CATE Estimation**
   - **DRLearner** (EconML): Doubly-robust with cross-fitting
   - **Causal Forest** (EconML `CausalForestDML`): forest on orthogonalised residuals
   - **X-Learner**: Two-stage meta-learner with propensity weighting

5. **Balance Diagnostics**
   - Standardized Mean Difference (SMD) before/after weighting
   - Love plots (target: SMD < 0.1)
   - Variance ratios, positivity checks

6. **Policy Evaluation**
   - **Qini Curve**: Cumulative gain vs. fraction targeted
   - **AUUC**: Area under uplift curve (normalized by random baseline)
   - Budget-constrained targeting: ROI, net benefit at different budgets
   - Segment-level uplift analysis

7. **Robustness Checks**
   - DoWhy causal graph and refutation tests
   - Placebo outcome tests
   - Leave-one-out covariate analysis
   - Sensitivity to unobserved confounding

### Key Metrics

- **ATE**: Average increase in outcome (e.g., spend) due to treatment
- **CATE**: Customer-level predicted treatment effect
- **AUUC**: Policy performance metric (higher = better targeting)
- **SMD**: Covariate balance metric (target < 0.1)
- **ROI**: Return on investment for targeting policy

## ⚠️ Assumptions & Limitations

### Causal Assumptions

1. **SUTVA** (Stable Unit Treatment Value Assumption)
   - No interference between units
   - No hidden versions of treatment

2. **Ignorability** (Selection on Observables)
   - All confounders are observed and included in X
   - **Limitation**: Cannot account for unobserved confounders

3. **Positivity** (Common Support)
   - All units have some probability of receiving treatment/control
   - Checked via propensity score overlap diagnostics

### Practical Limitations

- **Unobserved Confounding**: If important confounders are missing, estimates may be biased
- **Model Misspecification**: CATE estimates depend on correct functional form
- **External Validity**: Results may not generalize to different populations/time periods
- **Stationarity**: Assumes treatment effect is stable over time

### Recommended Validation

1. **Synthetic Data**: Verify ATE/CATE recovery on known ground truth
2. **Placebo Tests**: Check for non-effects where none should exist
3. **Refutation**: Use DoWhy to test robustness
4. **A/B Test**: Deploy learned policy in experiment to validate uplift

## 📊 Results

### Synthetic Data Validation

All numbers below come from `make all` with `config/default.yaml` (10,000 customers,
seed 42) and are written to `reports/tables/`. The generator's realised true ATE for this
seed is **$15.56** (the `true_ate: 15.0` config value is the base effect before the
heterogeneity terms).

| Method | ATE Estimate | 95% CI | Error vs true |
|--------|--------------|--------|---------------|
| Naive | $23.61 | [$22.92, $24.25] | **+$8.05** (biased by confounding) |
| IPW | $15.71 | [$15.17, $16.22] | +$0.15 |
| AIPW (Linear) | $15.79 | [$15.40, $16.26] | +$0.23 |
| AIPW (RF) | $15.99 | [$15.68, $16.56] | +$0.43 |

- **Naive** overstates the effect by about 50% because loyal, high-spending customers
  are more likely to receive coupons.
- **IPW and AIPW** land within a few percent of the truth, and every interval contains it.
- **Balance**: mean standardised mean difference drops from 0.100 to 0.011 after
  weighting; the largest imbalance (high-value segment, 0.47) drops to 0.01, the largest
  remaining SMD is 0.04, and all 20 covariates are below the 0.1 threshold.
- **CATE**: in-sample correlation with the true per-customer effect is 0.85 for the
  Causal Forest, 0.83 for DRLearner and 0.77 for the X-Learner. Per-segment predicted
  uplift tracks the truth closely (new customers $23.0 vs $23.5 true; low value $20.1
  vs $19.4; mid value $11.5 vs $11.4; high value $11.5 vs $11.3).
- **Policy**: normalised AUUC of 0.11 for Causal Forest targeting; the top 20% by
  predicted uplift have a predicted average effect of $22.84 against a population
  average of $15.56.

The CATE numbers above are in-sample; held-out CATE evaluation and wiring the DoWhy
refutation tests into the pipeline are planned follow-ups.

### Example Output

```
AVERAGE TREATMENT EFFECT ESTIMATION
====================================
[1/4] Naive difference in means...
Bootstrap CI (95%) from 100 iterations: $23.61 [22.92, 24.25]
[2/4] Inverse Propensity Weighting (IPW)...
Bootstrap CI (95%) from 100 iterations: $15.71 [15.17, 16.22]
[3/4] AIPW with linear outcome models...
Bootstrap CI (95%) from 100 iterations: $15.79 [15.40, 16.26]
[4/4] AIPW with Random Forest outcome models...
Bootstrap CI (95%) from 100 iterations: $15.99 [15.68, 16.56]

CATE ESTIMATION
===============
DRLearner CATE     - mean: $15.88, std: $5.43
Causal Forest CATE - mean: $15.78, std: $5.01
X-Learner CATE     - mean: $16.08, std: $5.63

BALANCE DIAGNOSTICS
===================
Before weighting - Mean SMD: 0.100   (70% of covariates below 0.1)
After weighting  - Mean SMD: 0.011   (100% of covariates below 0.1)

POLICY EVALUATION
=================
Area Under Uplift Curve (AUUC, normalised): 0.110
```

## 🔗 References

### Causal Inference Methods

- **Doubly-Robust Estimation**: Robins, Rotnitzky, Zhao (1994)
- **Causal Forest**: Wager & Athey (2018), "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"; Athey, Tibshirani & Wager (2019), "Generalized Random Forests"
- **DRLearner**: Chernozhukov et al. (2018), "Double/debiased machine learning for treatment and structural parameters"
- **X-Learner**: Künzel et al. (2019), "Metalearners for estimating heterogeneous treatment effects"

### Uplift Modeling

- **Qini Curve**: Radcliffe (2007), "Using control groups to target on predicted lift"
- **AUUC**: Gutierrez & Gérardy (2017), "Causal Inference and Uplift Modelling: A Review of the Literature"

### Software

- **EconML**: Microsoft Research causal ML library - [github.com/microsoft/EconML](https://github.com/microsoft/EconML)
- **DoWhy**: Microsoft causal inference framework - [github.com/microsoft/dowhy](https://github.com/microsoft/dowhy)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `make test && make lint`
5. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with**: EconML, DoWhy, scikit-learn, Streamlit
