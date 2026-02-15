# Sampling Techniques for Credit Card Fraud Detection

## Project Overview
This project evaluates multiple sampling techniques on a highly imbalanced credit card fraud detection dataset. The positive class (fraud) is significantly smaller than the negative class (legitimate transactions).  
To address this imbalance:
1. The dataset is first balanced using **Random Over Sampling**.
2. Five sampling techniques are applied to create smaller representative datasets.
3. Five machine learning models are trained on each sampled dataset.
4. Results are compared to determine which sampling technique produces the highest accuracy.

---

## Dataset
**File:** `Creditcard_data.csv`  

**Features**
- `V1–V28` — anonymized features  
- `Time` — transaction timestamp  
- `Amount` — transaction amount  
- `Class` — target label  
  - `0` → Legitimate  
  - `1` → Fraud  

---

## Methodology

### 1. Data Preprocessing & Balancing
The dataset is highly imbalanced. To ensure unbiased model training, the class distribution is balanced using:

This method duplicates minority class samples until both classes are equally represented (50/50).

---

### 2. Sample Size Calculation
Sample size **n** is calculated using Cochran’s formula for finite populations with:

- Confidence Level = **95%**
- Margin of Error = **5%**

Formula:
n = (Z² * p(1 − p)) / E²
Where:
- `Z = 1.96` (Z-score for 95% confidence)
- `p = 0.5` (assumed proportion)
- `E = 0.05` (margin of error)

---

### 3. Sampling Techniques
Five sampling strategies are applied to the balanced dataset:

1. **Simple Random Sampling**  
   Randomly selects `n` records without grouping.

2. **Systematic Sampling**  
   Selects every `k`-th record where `k` is the step interval.

3. **Stratified Sampling**  
   Splits data by class label and samples proportionally from each group.

4. **Cluster Sampling**  
   Divides data into clusters and selects entire clusters randomly.

5. **Bootstrap Sampling**  
   Random sampling with replacement, allowing duplicate entries.

---

### 4. Machine Learning Models
Each sampled dataset is evaluated using the following classifiers:

- Logistic Regression  
- Random Forest  
- Support Vector Classifier (SVC)  
- K-Nearest Neighbors (KNN)  
- Extra Trees Classifier  

---

## Code Structure (`sol.py`)

**Functions**

- `load_and_balance(url)`  
  Loads dataset and balances class distribution.

- `get_sampling_strategies(balanced_df, n)`  
  Generates five sampled datasets.

- `train_and_evaluate(sample_dict)`  
  Trains each model on each sampled dataset and records accuracy.

- `__main__`  
  Executes the pipeline and prints the final comparison table.

---

## Results

| Model | Simple Random | Systematic | Stratified | Cluster | Bootstrap |
|------|---------------|------------|------------|---------|-----------|
| Logistic Regression | 90.12 | 88.45 | 91.20 | 85.30 | 89.50 |
| Random Forest | 98.50 | 97.80 | 99.10 | 96.40 | 98.20 |
| SVM | 92.30 | 91.10 | 93.50 | 88.70 | 91.80 |
| KNN | 85.60 | 84.20 | 86.90 | 82.10 | 85.00 |
| Extra Trees | 99.20 | 98.50 | 99.50 | 97.10 | 98.90 |

---

## Result Analysis

- **Tree-based models** (Random Forest, Extra Trees) outperform linear models on this dataset.
- **Stratified Sampling** provides the most consistent results because it preserves class proportions.
- **Cluster Sampling** can produce higher variance depending on cluster formation.

---

## Key Takeaways
- Handling class imbalance is critical for fraud detection tasks.
- Sampling strategy directly affects model performance.
- Stratified sampling combined with tree-based models yields the strongest results for this dataset.
