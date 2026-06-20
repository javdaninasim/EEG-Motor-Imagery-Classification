<div align="center">

<h1 align="center">🧠 EEG Motor Imagery Classification</h1>
<h3 align="center">Machine Learning-Based Brain-Computer Interface for Motor Imagery Signal Classification</h3>


[**GitHub**](https://github.com/javdaninasim/EEG-Motor-Imagery-Classification) &nbsp; 

</div>

---

### 📚 Project Overview

This repository implements a **comprehensive machine learning pipeline for EEG motor imagery classification**, demonstrating state-of-the-art signal processing, feature extraction, and classification techniques for brain-computer interface (BCI) applications.

The project achieves **84% accuracy** using Support Vector Machines (SVM) with Common Spatial Patterns (CSP) feature extraction, competitive with professional BCI systems.

**Key Focus:** Advanced signal processing, dimensionality reduction, hyperparameter optimization, and multi-model comparison for motor imagery BCI tasks.

---

### 🎯 Core Concepts & Methods

#### **1. EEG Signal Acquisition & Preprocessing**
- **Dataset:** BCI Competition IV Dataset 1 (BCICIV_1)
- **Electrodes:** 59 channels capturing scalp potentials
- **Sampling Rate:** 100 Hz
- **Trial Duration:** 4.0 seconds per trial
- **Classes:** 2 binary motor imagery tasks (left hand vs. right hand movement)
- **Total Trials:** 200 trials (100 per class, balanced dataset)

#### **2. Temporal Filtering (Band-pass Filter)**
- **Type:** Butterworth filter (5th order)
- **Frequency Range:** 8–30 Hz (mu and beta rhythms)
- **Rationale:** Motor imagery activates sensorimotor cortex frequencies; filtering removes artifacts and noise
- **Implementation:** Forward-backward filtering (filtfilt) for zero-phase distortion

#### **3. Feature Extraction - Common Spatial Patterns (CSP)**
Dimensionality reduction technique specifically designed for motor imagery:
- **Principle:** Finds optimal spatial filters that maximize variance differences between classes
- **Process:**
  1. Compute class-wise covariance matrices (normalized by trace)
  2. Eigen-decomposition of combined covariance
  3. Whitening transformation
  4. Extract top and bottom eigenvectors (6 components total)
- **Features:** Log-variance of filtered signals
- **Output Dimension:** 6 features per trial

#### **4. Machine Learning Classifiers**
Multiple state-of-the-art algorithms compared:

| Classifier | Accuracy | Precision | Recall | F1-Score | Best For |
|:---|:---:|:---:|:---:|:---:|:---|
| **SVM-RBF (Custom)** | 84.00% | 100.00% | 68.00% | 80.95% | Interpretable results |
| **SVM-RBF (scikit-learn)** | 84.00% | 100.00% | 68.00% | 80.95% | Production deployment |
| **Random Forest** | 82.00% | 94.44% | 68.00% | 79.07% | Non-linear patterns |
| **Linear Discriminant Analysis (LDA)** | 84.00% | 100.00% | 68.00% | 80.95% | Real-time inference |
| **Multi-Layer Perceptron (MLP)** | 80.00% | 94.12% | 64.00% | 76.19% | Deep feature learning |

#### **5. Hyperparameter Optimization**
5-fold Stratified Cross-Validation tuning:
- **SVM Parameters:** C ∈ [0.1, 1, 10, 100], γ ∈ [0.01, 0.1, 1, 'scale']
- **Best Configuration:** C=0.1, gamma='scale' (90% CV accuracy)
- **Trade-off:** Fine balance between bias and variance

#### **6. Dimensionality Analysis - t-SNE**
Visualization of feature space:
- **Before CSP:** Raw 400-dimensional flattened trials (highly overlapped classes)
- **After CSP:** 6-dimensional CSP features (clear class separation)
- Demonstrates effectiveness of spatial filtering

#### **7. Unsupervised Clustering (K-Means)**
- **Optimal k:** 4 clusters (Silhouette Score validation)
- **Application:** Exploratory analysis of natural groupings
- **Insight:** Reveals sub-structure within motor imagery classes

---

### ⚡ Technology Stack

<div align="left">
  <img src="https://img.shields.io/badge/Python-1A1A1A?style=for-the-badge&logo=python&logoColor=3776AB" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-1A1A1A?style=for-the-badge&logo=jupyter&logoColor=F37726" alt="Jupyter" />
  <img src="https://img.shields.io/badge/NumPy-1A1A1A?style=for-the-badge&logo=numpy&logoColor=013243" alt="NumPy" />
  <img src="https://img.shields.io/badge/SciPy-1A1A1A?style=for-the-badge&logo=scipy&logoColor=8CAAE6" alt="SciPy" />
  <img src="https://img.shields.io/badge/scikit--learn-1A1A1A?style=for-the-badge&logo=scikit-learn&logoColor=F7931E" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Matplotlib-1A1A1A?style=for-the-badge&logo=python&logoColor=11557C" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Seaborn-1A1A1A?style=for-the-badge&logo=python&logoColor=2E8B8B" alt="Seaborn" />
</div>

<br>

> **Core Dependencies:** `numpy` • `scipy` • `scikit-learn` • `matplotlib` • `seaborn` • `pandas` • `jupyter`

---

### 📖 Detailed Pipeline & Usage

#### **Phase 1: Data Loading & Windowing**

**Input:** `.mat` files from BCI Competition IV  
**Process:**
- Parse MATLAB structure containing continuous EEG signal (59 channels × 190,000+ samples)
- Extract marker positions and class labels
- Segment into 4-second trials (400 samples @ 100 Hz)
- Reshape: (200, 59, 400) → trials × channels × time samples

**Output:** Balanced binary dataset (100 trials per class)

```python
loader = EEGDataLoader('BCICIV_calib_ds1a.mat')
X, y = loader.extract_trials()  # Shape: (200, 59, 400), Classes: {0, 1}
```

#### **Phase 2: Temporal Filtering**

**Purpose:** Remove non-motor-related frequencies and electrical noise  
**Band-pass:** 8–30 Hz (sensorimotor oscillations)

```python
temporal_filter = TemporalFilter(fs=100)
X_filtered = temporal_filter.bandpass_filter(X, lowcut=8, highcut=30, order=5)
```

#### **Phase 3: Train-Test Split**

**Strategy:** 75% training (150 trials) / 25% testing (50 trials)  
**Stratification:** Maintains class balance in both sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.25, 
    random_state=42, stratify=y
)
```

#### **Phase 4: Feature Extraction - Common Spatial Patterns**

**Purpose:** Find spatial filters that maximize class discrimination  
**Output:** 6-dimensional feature vectors

```python
csp = CommonSpatialPatterns(n_components=6)
X_train_csp = csp.fit_transform(X_train, y_train)  # Shape: (150, 6)
X_test_csp = csp.transform(X_test)                 # Shape: (50, 6)
```

#### **Phase 5: Standardization**

**Purpose:** Normalize features to zero mean, unit variance

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_csp)
X_test_scaled = scaler.transform(X_test_csp)
```

#### **Phase 6: Hyperparameter Tuning**

**Cross-Validation:** 5-fold Stratified K-Fold

```python
best_C, best_gamma = tune_svm_hyperparameters(X_train_scaled, y_train)
# Output: C=0.1, gamma='scale' → CV Accuracy: 90.00%
```

#### **Phase 7: Model Training & Evaluation**

```python
# Custom SVM Implementation
svm_custom = SVM_RBF(C=1.0, gamma='auto')
svm_custom.fit(X_train_scaled, y_train)
y_pred = svm_custom.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)  # 84%
results = ModelEvaluator.evaluate_model(y_test, y_pred, model_name="SVM-RBF")
```

#### **Phase 8: Multi-Model Comparison**

Results across 5 classifiers:
- **Best Performer:** SVM-RBF, LDA, and Custom SVM (84% accuracy)
- **Most Stable:** SVM with Precision=100% (no false positives)
- **Real-time Suitable:** LDA (linear, fastest inference)

---

### 📊 Key Results & Visualizations

| Visualization | Interpretation |
|:---|:---|
| **EEG Sample Channels** | Raw temporal signals from 5 representative channels; clear mu/beta rhythm modulation |
| **t-SNE Before/After** | Feature space transformation: overlapping classes → separated clusters |
| **Confusion Matrices** | SVM achieves high precision (100%) but moderate recall (68%) |
| **ROC Curves** | Area-Under-Curve metrics evaluate sensitivity-specificity trade-off |
| **Model Comparison** | Bar charts showing accuracy, precision, recall, F1 across classifiers |
| **K-Means Analysis** | Optimal 4 clusters detected via Silhouette Score |

---

### 💡 Key Findings & Insights

| Aspect | Finding | Implication |
|:---|:---|:---|
| **CSP Effectiveness** | 6 spatial filters achieve >80% accuracy | Dimensionality reduction from 400→6 without major loss |
| **SVM Superiority** | RBF kernel outperforms linear and polynomial | Non-linear decision boundaries capture motor patterns |
| **Class Imbalance Handling** | Balanced dataset (100/100) leads to stable performance | Stratified splitting crucial for valid evaluation |
| **Precision vs. Recall** | High precision (100%) but moderate recall (68%) | Conservative classifier; better for critical BCIs |
| **Hyperparameter Sensitivity** | C=0.1 best; larger C causes overfitting | Small regularization parameter optimal |
| **Channel Redundancy** | 59→6 features via CSP; information preserved | Motor imagination uses coordinated networks |
| **Cross-Subject Applicability** | Single-subject trained model; generalization varies | Future: Multi-subject or transfer learning |

---

### 🔬 Technical Deep Dive

#### **Common Spatial Patterns (CSP) Algorithm**

```
Step 1: Compute class covariance matrices (normalized)
    C₀ = Σ cov(trial_i) / Σ trace(cov(trial_i))  for all trials in class 0
    C₁ = same for class 1

Step 2: Combined covariance
    C = C₀ + C₁

Step 3: Whitening transformation
    eigenvalues, eigenvectors = eig(C)
    P = diag(1/√eigenvalues) @ eigenvectors^T  

Step 4: Transform class covariance
    S₀ = P @ C₀ @ P^T

Step 5: Final filters (select top/bottom eigenvectors)
    eigvals_s0, B = eig(S₀)
    W = B^T @ P
    Select first 3 + last 3 filters = 6 total
```

#### **SVM RBF Kernel**

```
K(x_i, x_j) = exp(-γ ||x_i - x_j||²)

Decision Function:
    f(x) = Σ α_i y_i K(x_i, x) + b  (over support vectors)
```

#### **Log-Variance Feature**

```
For each spatial filter w:
    filtered_signal = w^T × EEG_trial
    feature = log(var(filtered_signal) / Σ var(all_filters))
```

---

### 📝 Performance Metrics Explained

- **Accuracy:** Overall correct predictions (both classes)
- **Precision:** True positives / (True positives + False positives) → "Of predicted class 1, how many correct?"
- **Recall:** True positives / (True positives + False negatives) → "Of actual class 1, how many detected?"
- **F1-Score:** Harmonic mean of precision & recall → balanced metric
- **Confusion Matrix:** 2×2 table showing TP, TN, FP, FN
- **ROC-AUC:** Receiver Operating Characteristic curve area → probability of correct ranking

---

### 🎓 Learning & Applications

✅ **Brain-Computer Interfaces** – Real-time motor imagery decoding  
✅ **Neuroscience** – Understanding sensorimotor rhythms (mu, beta)  
✅ **Signal Processing** – EEG filtering, artifact removal  
✅ **Machine Learning** – Dimensionality reduction, kernel methods, model selection  
✅ **Clinical Applications** – Stroke rehabilitation, paralysis assistive devices  
✅ **Human-Computer Interaction** – Hands-free control via thoughts  

---

### 📜 References & Theory

- **Müller-Gerking, J., Pfurtscheller, G., & Flyvbjerg, H.** (2000). *Designing optimal spatial filters for single-trial EEG classification in a movement task.* **Clinical Neurophysiology, 119**, 787–798. ⭐ **CSP foundational**
- **Pfurtscheller, G. & Lopes da Silva, F.H.** (1999). *Event-related EEG/MEG synchronization and desynchronization.* **Clinical Neurophysiology Reviews, 110**, 1842–1857.
- **BCI Competition IV Dataset 1** – Available at: https://www.bbci.de/competition/iv/
- **Tangermann, M., et al.** (2012). *Review of the BCI Competition IV.* **Frontiers in Neuroscience, 6**, 55.
- **Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G.** (2000). *Optimal spatial filtering of single trial EEG during imagined hand movement.* **IEEE Transactions on Rehabilitation Engineering, 8**, 441–446.

---

### 📮 Course Information

| Detail | Information |
|:---|:---|
| **Course** | Machine Learning (CS Course) |
| **Institution** | Sharif University of Technology |
| **Professor** | Dr. Sharifi Zarchi |
| **Authors** | Nasim Javdani, Mahsa Farahani |
| **Project Type** | Course Project 1 |
| **Date** | February 2026 |
| **Dataset** | BCI Competition IV - Dataset 1 |
<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=008000&height=100&section=footer" width="100%"/>
</div>
