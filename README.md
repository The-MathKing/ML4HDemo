# SpotGuard: Conformal Gatekeeping for Spatial Transcriptomic Deconvolution

[![GitHub Pages Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-purple?style=flat-square&logo=github)](https://The-MathKing.github.io/ML4HDemo/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Unit Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen?style=flat-square)](tests/)

**SpotGuard** is a geometry-aware conformal gatekeeping framework and interactive clinical demonstration tool for spatial transcriptomic (ST) deconvolution under atlas-to-tissue distribution shift.

---

## 🌟 Interactive Live Demo

Experience the live interactive demo directly in your browser:  
👉 **[https://The-MathKing.github.io/ML4HDemo/](https://The-MathKing.github.io/ML4HDemo/)**

- **Dynamic $\alpha$-Slider**: Modulate the target miscoverage rate $\alpha \in [0.01, 0.30]$ in real-time.
- **Conformal Reliability Masking**: Watch uncalibrated tumor-stroma boundary spots get dynamically gated out.
- **Simplex Ternary Inspector**: Click any tissue spot to inspect its compositional uncertainty ball and covariate shift weight.

---

## 🔬 Core Methodology

1. **Total Variation Nonconformity on the Closed Simplex**:
   $$S_{\text{TV}}(x, y) = \frac{1}{2} \|y - \hat{y}\|_1 = \frac{1}{2} \sum_{k=1}^K |y_k - \hat{y}_k|$$
   - Fully defined on the closed simplex $[0, 1]^K$, eliminating zero-singularity and $\epsilon$-regularization instabilities.
   - Bounded strictly in $[0, 1]$ with direct clinical interpretability: at confidence $1-\alpha$, no more than $q_\alpha \times 100\%$ of cell proportions are misallocated.

2. **Covariate Shift Reweighting & Kish's ESS**:
   - Trains a regularized domain discriminator $d_\phi(x)$ on spot gene embeddings to estimate density-ratio weights $w(x) = \frac{d_\phi(x)}{1 - d_\phi(x)} \frac{M_{\text{ref}}}{M_{\text{query}}}$.
   - Computes Effective Sample Size $\text{ESS} = \frac{(\sum w)^2}{\sum w^2}$ with percentile clipping to guard against weight explosion.

3. **Discriminator Realism Gate**:
   - Verifies that synthetic pseudo-spots match real tissue characteristics with an optimal realism window: $0.60 \le \text{AUC}(d_\phi) \le 0.80$.

4. **Decision-Ambiguity Gating**:
   - Gates spots strictly when uncertainty intervals straddle a clinical threshold (e.g. CD8$^+$ infiltration $> 5\%$), avoiding arbitrary volume thresholds.

---

## 📊 Empirical Benchmarks

Summary of empirical coverage across test regimes (Nominal Target: 90%):

| Regime | Method | Empirical Coverage | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: |
| **In-Distribution** | Bayesian Joint Credible Region | 1.0% | N/A |
| **In-Distribution** | **SpotGuard TV Conformal** | **92.4%** | [90.2%, 94.6%] |
| **Shifted (Cross-Donor)** | Bayesian Joint Credible Region | 2.0% | N/A |
| **Shifted (Cross-Donor)** | Unweighted Conformal | 89.8% | N/A |
| **Shifted (Cross-Donor)** | **SpotGuard Weighted** | **89.8%** | [87.4%, 92.4%] |

- **Selective Prediction AURC**: `0.1381`
- **Total Variation Bound ($q_\alpha$)**: `0.245`

---

## 🚀 Quickstart & Reproduction

### 1. Run Unit Tests
```bash
python3 -m unittest discover -s tests -v
```

### 2. Run the End-to-End Benchmark Pipeline
```bash
python3 run_benchmarks.py
```

### 3. Launch the Local Interactive Web Application
```bash
python3 app.py
# Open http://localhost:8000 in your browser
```

### 4. Compile the Spec Sheet Manuscript
```bash
tectonic main.tex
# Generates main.pdf
```

---

## 📁 Repository Structure

```
ML4HDemo/
├── index.html                # Static Web App for GitHub Pages
├── .nojekyll                 # GitHub Pages configuration
├── app.py                    # Local interactive demo server
├── run_benchmarks.py         # End-to-end benchmark execution script
├── benchmark_results.json    # Exported benchmark metrics
├── DESIGN.md                 # Methodological specification & blueprint
├── TARGETS.md                # Testable empirical hypotheses
├── main.tex                  # Spec Sheet LaTeX source
├── main.pdf                  # Compiled PDF document
├── tests/
│   └── test_conformal.py     # Unit test suite
└── spotguard/
    ├── __init__.py           # Package exports
    ├── engine.py             # Conformal mathematics & decision gating
    ├── data.py               # Data structures & paired benchmark generators
    ├── simulator.py          # Realistic pseudo-spot simulator & AUC gate
    ├── deconvolution.py      # Bayesian deconvolution & posterior sampling
    └── evaluate.py           # Evaluation harness & Risk-Coverage curves
```

---

## 📄 License & Attribution

Author: **Aryan Padarthi**  
Licensed under the [MIT License](LICENSE).
