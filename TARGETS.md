# SpotGuard: Empirical Hypotheses & Target Benchmarks

> **Note**: These values represent testable empirical hypotheses and evaluation targets to be validated by the experimental pipeline.

---

## Target 1: Marginal & Conditional Coverage Under Shift
Target evaluation on public benchmark pairs (e.g. Visium HER2+ / Slide-seqV2 mouse cortex):

| Regime | Baseline (Bayesian Posterior) Nominal 90% | SpotGuard Target Coverage | Target Simplex TV Bound ($q_\alpha$) |
| :--- | :--- | :--- | :--- |
| **In-Distribution** | $\approx 88 - 92\%$ | $90 \pm 1.5\%$ | $\le 0.10$ |
| **Cross-Donor Shift** | Degrades to $\approx 65 - 75\%$ | $90 \pm 1.5\%$ | $\le 0.18$ |
| **Cross-Platform Shift** | Degrades to $\approx 50 - 65\%$ | $90 \pm 2.0\%$ | $\le 0.25$ |

---

## Target 2: Weight Effective Sample Size (ESS)
- Target: $\text{ESS} \ge 15\%$ of calibration set size $M$ after weight clipping.
- Discriminator validation: $0.60 \le \text{AUC}(d_\phi) \le 0.80$ on synthetic vs. real spot representations.

---

## Target 3: Decision Ambiguity Retention
- Target retention rate at $\alpha = 0.10$: $\ge 70\%$ of tissue spots retained with unambiguous clinical decisions (e.g. CD8$^+$ infiltration $> 5\%$).
- Selective deconvolution: Monotonic risk reduction under selective coverage ordering.
