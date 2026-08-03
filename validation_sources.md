# Validation Rule Sources

Justifications and references for the thresholds in `validation.py`.

---

## 1. n ≥ 20 (minimum absolute sample size)

**Support: Moderate. Conservative; works for variance-covariance methods but n=20 is quite small.**

The sample mean and variance-covariance method performs acceptably on small samples when data are well-behaved, but simulation studies treat n=20 as a "very small sample." More robust estimators (MCD, MVE) are preferred at small n.

- Brereton, R. G. (2015). *The Mahalanobis distance and its relationship to principal component scores.* Journal of Chemometrics, 29(3), 143–145. https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/cem.2692
- Brereton, R. G. (2021). *Mahalanobis distances for ecological niche modelling and outlier detection: implications of sample size, error, and bias for selecting and parameterising a multivariate location and scatter method.* PeerJ, 9, e11436. https://pubmed.ncbi.nlm.nih.gov/34026369/
- Nath, R., & Pavur, R. (1985). *On the bias of Mahalanobis distance due to limited sample size effect.* IEEE Transactions, https://ieeexplore.ieee.org/document/395756/

---

## 2. n > k (rows must exceed variables)

**Support: Strong — mathematical requirement, not a heuristic.**

When n ≤ k the sample covariance matrix is rank-deficient and singular; matrix inversion (required for Mahalanobis distance) is impossible. Even when n is only slightly above k, the eigenstructure is distorted and estimates are unreliable.

- Wikipedia: Estimation of covariance matrices. https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
- Fan, J., Liao, Y., & Liu, H. (2016). *An overview on the estimation of large covariance and precision matrices.* The Econometrics Journal, 19(1), C1–C32. https://arxiv.org/pdf/1504.02995

---

## 3. n/k ≥ 5 (hard minimum ratio)

**Support: Strong. Widely cited minimum in multivariate statistics textbooks.**

Hair et al. (2018) state that the absolute minimum acceptable ratio is 5 observations per variable for stable covariance estimation and regression-class analyses. Below this, the covariance matrix is unreliable regardless of absolute n.

- Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2018). *Multivariate Data Analysis* (8th ed.). Cengage Learning.
- Tabachnick, B. G., & Fidell, L. S. (2013). *Using Multivariate Statistics* (6th ed.). Pearson. https://www.pearsonhighered.com/assets/preface/0/1/3/4/0134790545.pdf

---

## 4. n/k ≥ 10 (preferred ratio)

**Support: Strong. Widely cited preferred guideline; not a hard minimum.**

The 10:1 ratio is a standard rule of thumb originating with Nunnally (1978) and repeated across the multivariate statistics literature. Moons et al. (2009) recommend at least 10 observations per predictor for multivariate analyses. It assumes moderate effect sizes and low multicollinearity but does not carry a formal power guarantee.

- Nunnally, J. C. (1978). *Psychometric Theory* (2nd ed.). McGraw-Hill.
- Moons, K. G. M., et al. (2009). *Prognosis and prognostic research: what, why, and how?* BMJ, 338, b375.
- Vittinghoff, E., & McCulloch, C. E. (2007). *Relaxing the rule of ten events per variable in logistic and Cox regression.* American Journal of Epidemiology, 165(6), 710–718. https://pmc.ncbi.nlm.nih.gov/articles/PMC6519266/
- MacCallum, R. C., Widaman, K. F., Zhang, S., & Hong, S. (1999). Sample size in factor analysis. *Psychological Methods*, 4(1), 84–99. https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/RatCaseVar

---

## 5. n ≥ 30 (chi-square approximation accuracy)

**Support: Moderate. Reasonable but somewhat liberal for chi-square specifically; n ≥ 50 is better.**

Under multivariate normality, squared Mahalanobis distances follow a chi-square distribution with k degrees of freedom. The n ≥ 30 threshold derives from the Central Limit Theorem heuristic for normal approximation of means. For chi-square approximation specifically, n ≥ 50 is more commonly cited. For small k (2–10 variables), n ≥ 30 provides acceptable accuracy in practice.

- Penn State STAT 504: Normal and Chi-Square Approximations. https://dev.stat.vmhost.psu.edu/stat504/lesson/2/2.1
- Chi-squared distribution — Wikipedia. https://en.wikipedia.org/wiki/Chi-squared_distribution
- ScienceDirect Topics: Mahalanobis Distance. https://www.sciencedirect.com/topics/mathematics/mahalanobis-distance

---

## 6. Max 30% missing per column

**Support: Moderate. Empirically grounded upper bound from sparse covariance literature.**

No standard statistical source sets a precise per-column threshold, but sparse covariance estimation literature shows methods remain competitive when column-wise missingness stays below ~30%; above that, individual variance and covariance estimates are increasingly unreliable. The 30% value is more defensible than the commonly-used 40% heuristic.

- Fan, R., & Shen, X. (2020). *High-dimensional covariance matrix estimation with missing and dependent data.* University of Michigan. https://deepblue.lib.umich.edu/bitstream/handle/2027.42/163035/rogerfan_1.pdf
- Loh, P.-L., & Wainwright, M. J. (2012). *Structure estimation for discrete graphical models: Generalized covariance matrix estimation and its alternatives.* https://arxiv.org/abs/0903.5463

---

## 7. Max 20% overall missingness

**Support: Strong. Well-supported by Allison and modern missing-data practice.**

Allison (2002) identifies 20% as the practical threshold at which multiple imputation becomes recommended over simpler approaches. Analyses based on covariance estimation (including Mahalanobis distance) should flag overall missingness above this level because it can systematically bias the covariance structure under non-MCAR patterns.

- Allison, P. D. (2002). *Missing Data.* Sage Publications. https://statisticalhorizons.com/wp-content/uploads/Allison_MissingData_Handbook.pdf
- Sterne, J. A. C., et al. (2009). *Multiple imputation for missing data in epidemiological and clinical research: potential and pitfalls.* BMJ, 338, b2393. https://pmc.ncbi.nlm.nih.gov/articles/PMC3701793/
- Austin, P. C., et al. (2021). *Evaluation of multiple imputation with large proportions of missing data.* BMC Medical Research Methodology, 21, 197. https://pmc.ncbi.nlm.nih.gov/articles/PMC8426774/

---

## 8. Condition number ≤ 1000 (numerical stability of matrix inversion)

**Support: Strong. Aligns with standard numerical analysis definition of ill-conditioning.**

The condition number κ(A) = λ_max / λ_min measures how much error in the input is amplified by matrix inversion. The rule of thumb is that log₁₀(κ) digits of precision are lost; κ = 10³ implies ~3 decimal places lost. Matrices with κ > 10³ are conventionally described as ill-conditioned. For covariance matrix inversion in Mahalanobis distance, this is an established hard threshold.

- Wikipedia: Condition number. https://en.wikipedia.org/wiki/Condition_number
- Won, J.-H., Lim, J., Kim, S.-J., & Rajaratnam, B. (2013). *Condition number regularized covariance estimation.* Journal of the Royal Statistical Society: Series B, 75(3), 427–450. https://pmc.ncbi.nlm.nih.gov/articles/PMC3667751/
- Uboldi, A., et al. (2020). *An efficient numerical method for condition number constrained covariance matrix approximation.* Signal Processing, 174, 107615. https://www.sciencedirect.com/science/article/abs/pii/S009630032030878X
- Pardo-Igúzquiza, E., & Dowd, P. A. (1997). *On the condition number of covariance matrices in kriging.* Mathematical Geology, 29(5), 573–577. https://link.springer.com/article/10.1007/BF02065878
