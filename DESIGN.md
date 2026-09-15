# SpotGuard: Design Document & Methodological Blueprint

> **Status**: Architectural Specification & Theoretical Design (Pre-execution Blueprint)  
> **Target Venues**: CHIL 2027 / ML4H 2027 / arXiv

---

## 1. Core Clinical Problem & Motivation

Spatial transcriptomics (ST) deconvolution methods (e.g., `cell2location`, `RCTD`) quantify cell-type fractions on $55\,\mu\text{m}$ tissue spots to delineate tumor-immune boundaries, map tertiary lymphoid structures (TLS), and measure cytotoxic CD8$^+$ T-cell infiltration. 

### Silent Failure Mode
When single-cell reference atlases diverge from clinical tissue specimens in donor genetics, tissue preservation (FFPE vs. fresh frozen), dissociation protocols, or sequencing platforms (e.g. 10x Chromium vs. Slide-seqV2), the exchangeability assumption is broken. Existing Bayesian posterior credible intervals and asymptotic standard errors suffer severe miscalibration:
- The model outputs high posterior certainty on spots where it is systematically wrong.
- Clinicians reading infiltration maps cannot distinguish high-trust from high-risk calls, risking false margin assessments.

---

## 2. Core Methodological Corrections

### M1. Nonconformity Metric: Total Variation on the Closed Simplex
Rather than centered log-ratio ($\clr$) transforms which are undefined at exact zero proportions (the majority of spatial deconvolution outputs), we define the primary nonconformity score via **Total Variation (TV)** distance:
$$S_{\text{TV}}(x, y) = \frac{1}{2} \|y - \hat{y}\|_1 = \frac{1}{2} \sum_{k=1}^K |y_k - \hat{y}_k|$$
- **Domain**: Fully defined everywhere on the closed probability simplex $\Delta^{K-1}$, including sparse boundaries.
- **Range**: Bounded strictly in $[0, 1]$.
- **Interpretability**: $S_{\text{TV}} \le q_\alpha$ guarantees that at confidence level $1-\alpha$, no more than $q_\alpha \times 100\%$ of the spot's total cellular composition is misallocated.

*(Aitchison distance with $\epsilon$-replacement is retained as an ablation and secondary compositional coordinate representation).*

### M2. Shift Formulation & Honesty
Cross-platform transfer is a combination of covariate shift $P(X)$ and technical/capture efficiency shift $P(Y \mid X)$. We formalize the method under weighted conformal prediction with explicit acknowledgment of approximate coverage under residual concept shift.

### M3. Pseudo-Spot Realism & Discriminator Validity Gate
Synthetic pseudo-spots $\mathcal{D}_{\text{cal}}$ must include:
1. Dirichlet mixture proportions $y \sim \text{Dirichlet}(\beta)$ with $\beta$ fitted to empirical deconvolution outputs.
2. Negative binomial count dispersion (not naive summation).
3. Ambient RNA profile contamination $\rho \sim \text{Beta}(a, b)$ estimated from empty droplets.
4. **The Simulator Realism Gate**: Train a discriminator $d_\phi(x)$ to separate pseudo-spots from real spots. We require:
   $$0.60 \le \text{AUC}(d_\phi) \le 0.80$$
   An $\text{AUC} > 0.85$ indicates an over-clean synthetic calibration set that collapses the effective sample size ($\text{ESS}$).

### M4. Decision-Ambiguity Gating (Replacing Arbitrary Volume Thresholds)
Given a clinical threshold function $d(y) = \mathbb{I}(y_{\text{CD8}^+} > \tau_{\text{hot}})$:
$$\mathcal{M}_\alpha(x) = \begin{cases} 
1 \text{ (Trust)}, & \text{if } \forall y, y' \in C_\alpha(x), \; d(y) = d(y') \\
0 \text{ (Mask/Ambiguous)}, & \text{if } \exists y, y' \in C_\alpha(x) \text{ s.t. } d(y) \neq d(y')
\end{cases}$$
This gates spots strictly when uncertainty spans a clinical decision threshold, rather than penalizing wide regions whose decisions are unanimous.

---

## 3. Required Reporting & Evaluation Metrics
1. **Empirical Coverage**: With spatial block bootstrap confidence intervals.
2. **Retention Rate vs. $\alpha$**: Proportion of the tissue section remaining unmasked at miscoverage level $\alpha$.
3. **Effective Sample Size ($\text{ESS}$)**: $\text{ESS} = \frac{(\sum w)^2}{\sum w^2}$ reported as a percentage of calibration set size $M$.
4. **Selective Prediction AURC**: Area Under the Risk-Coverage Curve across varying decision thresholds.

---

## 4. Execution Roadmap (8-Week Build)
- **Phase 1**: Public Data ETL (`GSE176078` breast atlas + Visium + Slide-seqV2 cortex).
- **Phase 2**: Noise-injected pseudo-spot simulator + Discriminator AUC gate.
- **Phase 3**: Cached `cell2location` and `RCTD` posterior sampling.
- **Phase 4**: Conformal engine (`scores.py`, `weights.py`, `conformal.py`, `gate.py`).
- **Phase 5**: Evaluation harness & ablation studies.
- **Phase 6**: Streamlit/Dash interactive artifact.
- **Phase 7**: Final 2-page spec sheet + verified 2-minute walkthrough video for CHIL/arXiv.
