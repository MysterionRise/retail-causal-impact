# Output Interpretation Guide for Non-Experts

Every figure and number here is produced by `make all` (default profile, 10,000 synthetic customers, seed 42) and lands in `reports/figures/` and `reports/tables/`. The dollar amounts are per customer unless a total is stated.

## The three questions the analysis answers

1. **Did the coupon campaign actually work?** Is the measured lift real or an artefact of who was targeted?
2. **Who benefits most from coupons?** Which customers should get the next one?
3. **How much budget should we spend?** What is the net benefit of different targeting rules?

## The four outputs to present

### 1. "Did the coupons work?" · `04_ate_comparison.png`

**What it shows.** Four estimators of the same effect with 95% intervals. The naive row is the plain treated-versus-control gap; the others adjust for who was targeted.

| Method | Estimate | 95% interval |
|--------|----------|--------------|
| Naive difference | $23.61 | $22.92 to $24.25 |
| IPW | $15.71 | $15.17 to $16.22 |
| AIPW (linear) | $15.79 | $15.40 to $16.26 |
| AIPW (random forest) | $15.99 | $15.68 to $16.56 |

The true effect in this synthetic campaign is **$15.56**.

**What to tell your boss.**
> "The raw comparison says a coupon adds $23.61 of spend. That is inflated because we sent coupons to our best customers, who spend more anyway. After correcting for that, the real effect is about $15.80 per coupon. We were overstating the impact by roughly 50%."

**Red flags.** A naive estimate far above the adjusted ones means the targeting was biased, as here. Intervals that cross zero mean the campaign may not be working.

### 2. "Who responds?" · `05_cate_distribution.png` and `07_segment_uplift.png`

**What they show.** The spread of predicted per-customer effects, and the average by segment.

| Segment | Coupon rate | Effect (doubly-robust) | True effect |
|---------|-------------|------------------------|-------------|
| New customers | 15% | $22.95 | $23.46 |
| Low value | 29% | $19.90 | $19.37 |
| Mid value | 27% | $11.70 | $11.38 |
| High value | 51% | $11.80 | $11.32 |

**What to tell your boss.**
> "New and low-value customers gain over $20 of spend from a coupon; high-value customers gain under $12, and they are the ones who got the most coupons. We have been spending where the effect is smallest."

### 3. "Who should we target?" · `08_targeting_ml_vs_causal.png`

**What it shows.** Given the same budget (the top 20% of 3,000 held-out customers), four targeting rules scored by incremental spend per targeted customer. The doubly-robust column is what you could compute on real data; the true column exists only because the data is synthetic.

| Targeting rule | Doubly-robust | True |
|----------------|---------------|------|
| Oracle (true effect) | $24.49 | $24.27 |
| Causal uplift model | $23.19 | $22.99 |
| Random | $15.41 | $15.59 |
| Predictive response model | $13.83 | $12.46 |

**What to tell your boss.**
> "A standard predictive model picks the customers most likely to spend. Those are loyal customers who would have spent anyway, so it does worse than picking at random. Ranking by predicted uplift instead yields $23 per targeted customer, about 1.7 times the predictive model."

**Red flags.** If the causal line in the left-hand chart hugs the random diagonal, the uplift model has not found real heterogeneity and targeting will not help.

### 4. "How much should we spend?" · the budget table in `policy_results.csv` and the dashboard what-if panel

**What it shows.** Net benefit of the uplift policy at increasing budget shares on the held-out set, at $1 per coupon.

| Budget share | Coupons | Incremental spend per coupon | Net benefit |
|--------------|---------|------------------------------|-------------|
| 10% | 300 | $22.47 | $6,440 |
| 20% | 600 | $23.19 | $13,312 |
| 30% | 900 | $22.65 | $19,482 |
| 50% | 1,500 | $20.95 | $29,926 |

Across the whole base of 10,000 customers, a 20% budget at $1 per coupon returns about $49,300 of incremental spend for $2,000 of coupons. At $1 the predicted effect never falls below the coupon cost, so the break-even share is 100%; raise the cost (the dashboard input, or ask the copilot "what if coupons cost $5") to find the point where marginal customers stop paying for themselves.

## Supporting outputs: can we trust this?

### `03_love_plot.png` · balance

Each row is a customer characteristic. Orange is the imbalance between treated and control before weighting, blue after. The mean standardized mean difference drops from 0.100 to 0.011 and all 20 covariates end up below the 0.1 line. Treated and control customers are now comparable on everything we measured.

### `02_propensity_distribution.png` · overlap

Treated and control customers must overlap in their probability of receiving a coupon. They do: no customer sits at the extremes, so every customer has a plausible counterpart in the other group.

### `09_cate_validation.png` · held-out calibration

Customers are binned by predicted effect. Predicted (blue), doubly-robust (orange with intervals) and true (black) effects rise together from about $10 in the lowest decile to about $23 in the highest. The Causal Forest's predictions correlate 0.84 with the true per-customer effect on customers it never saw during fitting.

### `10_refutation_checks.png` · refutation

Six checks that should either land on zero (placebo tests) or stay at the original estimate (stability tests). All six pass.

| Check | Expected | Re-estimated |
|-------|----------|--------------|
| Placebo treatment, DoWhy | $0 | $0.07 |
| Random common cause added, DoWhy | $15.21 | $15.21 |
| Random 80% subset, DoWhy | $15.21 | $15.22 |
| Placebo treatment on the AIPW estimate | $0 | $0.03 ± $0.23 |
| Effect on a pre-campaign variable | $0 | $0.01, interval covers zero |
| Drop the most influential covariate (loyalty) | $15.79 | $17.10 |

The robustness value is 0.51: an unobserved confounder would need to explain more than half of the remaining variance of both coupon assignment and spend to reduce the estimate to zero.

**What to tell your boss.**
> "We tried to break the result six ways and could not. A fake coupon shows no effect, the estimate does not move when we add noise or drop part of the data, and it would take an unmeasured factor stronger than loyalty itself to explain it away."

## A 10-minute presentation

1. **The bottom line** (`04_ate_comparison.png`): "Coupons add about $15.80 per customer, not the $23.61 the raw numbers suggest. We were overstating the effect by about half."
2. **The opportunity** (`08_targeting_ml_vs_causal.png`): "Targeting by predicted uplift returns $23 per targeted customer. Our spend-prediction model returns $14, less than random."
3. **Who to target** (`07_segment_uplift.png`): "New and low-value customers respond at over $20; high-value customers at under $12, and they currently receive half the coupons."
4. **Why trust it** (`10_refutation_checks.png`): "Six refutation checks pass. It would take an unmeasured factor explaining more than half the remaining variance to overturn the result."
5. **Next step**: pilot uplift-based targeting against the current rule in an A/B test and compare incremental spend per coupon.

## Common questions

**Why not just compare treated and control?** Because coupons were not assigned at random. Loyal customers both got more coupons and spent more, so the comparison mixes the coupon's effect with the customers' loyalty. The adjusted estimators separate the two.

**How confident are the numbers?** The intervals in the first table come from a bootstrap. On this synthetic campaign the adjusted estimates sit within about $0.40 of the truth, and the policy estimates sit within about a dollar of the true targeting value.

**Can I trust individual customer predictions?** The ranking is what matters and it is validated out of sample. Treat an individual prediction as a score, not a measurement.

**What can go wrong on real data?** A factor that drives both targeting and spend but is not in the data would bias the estimate; the robustness value says how strong it would have to be. Effects can shift over time or across markets, so re-estimate per campaign. The analysis assumes customers do not influence each other's response.

## Glossary

- **ATE**: average effect of a coupon across all customers.
- **CATE**: predicted effect for one customer given their characteristics.
- **Doubly-robust (AIPW)**: an estimator that combines a model of who gets coupons with a model of spend and stays consistent if either one is right.
- **SMD**: imbalance between groups on one characteristic; under 0.1 is balanced.
- **Backdoor set**: the variables that must be adjusted for, derived from the causal graph.
- **Robustness value**: how strong an unmeasured confounder would have to be to explain the effect away.
