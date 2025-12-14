# Healthcare Cost Prediction & Uncertainty Quantification

**Notice that this research was supervised by Dr Liu at LNUT. The dataset cannot be posted as open-source.**

## Project Overview
This project develops a robust machine learning pipeline to forecast hospital procedure costs. It addresses high-dimensional healthcare data challenges by integrating **Natural Language Processing (NLP)** for clinical descriptions and **Conformal Prediction** for reliable uncertainty quantification.

**Key Achievement:** The model outperformed linear baselines by **21% (MAE)** and achieved **89% coverage reliability** using conformal prediction intervals, ensuring safe deployment for budget planning.

## Key Methodology
1.  **Feature Engineering**: 
    * **NLP**: TF-IDF vectorization + Truncated SVD for high-dimensional diagnosis descriptions.
    * **Temporal**: Construction of Lag-1 and Year-over-Year growth features.
    * **Target Encoding**: Smoothed target encoding for high-cardinality facility IDs.
2.  **Modeling**:
    * **Baseline**: ElasticNet (Linear).
    * **Advanced**: LightGBM (Gradient Boosting) with Early Stopping.
3.  **Uncertainty Quantification**:
    * Implemented **Split Conformal Prediction** to generate distribution-free, statistically valid prediction intervals (Target $\alpha=0.1$).
4.  **Interpretability**:
    * SHAP (SHapley Additive exPlanations) analysis to identify cost drivers (e.g., Clinical Severity vs. Historical Trends).

## Results
* **Accuracy**: Reduced MAE from 8640 (Baseline) to 6831 (LightGBM).
* **Reliability**: Conformal Prediction achieved **88.92% coverage** on the test set (Target: 90%), correcting the under-confidence of standard Quantile Regression (65% coverage).

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
