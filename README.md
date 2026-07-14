# 🤖 SVM Model Analyzer (From Scratch)
### *Convex Optimization & Mathematical Programming Project*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B.svg)](https://streamlit.io/)
[![CVXOPT](https://img.shields.io/badge/CVXOPT-QP%20Solver-brightgreen.svg)](https://cvxopt.org/)
[![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-013243.svg)](https://numpy.org/)



## 📌 Project Overview

**SVM Model Analyzer** is a comprehensive, rigorous implementation of **Support Vector Machines (SVMs) built entirely from scratch in Python** using fundamental mathematical programming techniques. Instead of relying on pre-packaged classification estimators like `scikit-learn.svm.SVC`, this project directly solves the underlying **Primal and Dual Convex Optimization problems** using:

1. **Exact Quadratic Programming (QP)** via the `CVXOPT` solver.
2. **First-Order Primal Methods** including custom **Gradient Descent (GD)** and **Subgradient Descent** algorithms.

Accompanying the scratch implementations is an interactive **Streamlit Web Application** (`svm_app.py` / `fsvm_app.py`) that allows users to upload custom datasets, explore pairwise feature distributions, train linear and kernel SVMs, inspect support vectors, and compare optimization convergence curves in real time.

---

## 🧠 Mathematical & Algorithmic Implementations

### 1. 🔒 Hard Margin Linear SVM
Designed for perfectly linearly separable datasets by maximizing the separating hyperplanes' geometric margin without slack variables:
- **Optimization Formulation (Dual QP)**:
  $$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j x_i^T x_j$$
  $$\text{subject to } \sum_{i=1}^{n} \alpha_i y_i = 0, \quad \alpha_i \ge 0$$
- Solved using `cvxopt.solvers.qp`. Support vectors are identified where $\alpha_i > 10^{-5}$.

---

### 2. 🧈 Soft Margin Linear SVM
Designed to handle noisy or overlapping data by introducing a penalty parameter **$C$** and slack variables $\xi_i \ge 0$:
- **Primal Optimization (Hinge Loss Minimization)**:
  $$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \max\left(0, 1 - y_i(w^T x_i + b)\right)$$
- **Supported Solvers**:
  - **Quadratic Programming (QP)**: Dual QP with upper box constraints ($0 \le \alpha_i \le C$).
  - **Gradient Descent (GD)**: Iterative updates on primal weights with fixed learning rate $\eta$.
  - **Subgradient Descent**: Robust updates for non-differentiable hinge loss points with decaying step sizes $\eta_t = \frac{\eta_0}{\sqrt{t}}$.

---

### 3. 🌐 Non-Linear Kernel SVMs
Extends linear classification to complex non-linear decision boundaries via the **Kernel Trick** ($K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$):
- **Implemented Kernel Functions**:
  - **Radial Basis Function (RBF / Gaussian)**:
    $$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$
  - **Polynomial Kernel**:
    $$K(x_i, x_j) = (\gamma x_i^T x_j + c_0)^d$$
  - **Linear & Sigmoid Kernels**
- Supports training via **Dual QP**, **Stochastic Gradient Descent (SGD)**, and **Subgradient Optimization**.

---

## 🌟 Key Features of the Interactive Dashboard

- **📤 Dynamic Data Upload & Preprocessing**:
  - Upload CSV or Excel datasets (e.g., `data_banknote_authentication.csv`) or generate synthetic classification benchmarks on the fly.
  - Automatically standardizes features (`StandardScaler`) and encodes binary targets to $\{-1, +1\}$.
- **📊 Interactive Data Visualization**:
  - Inspect dataset summaries, class distribution pie charts, and custom pairwise 2D feature scatter plots.
- **🎯 Decision Boundary & Support Vector Visualization**:
  - Plot high-resolution 2D decision contours showing exact margin separations and highlight discovered support vectors.
- **📈 Optimization Diagnostics & Benchmarking Suite (`SVMAnalyzer`)**:
  - Real-time plots of **Loss Convergence Curves** across training iterations.
  - **Training vs. Validation Accuracy Curves**.
  - Direct comparative benchmarking across optimization algorithms (QP vs. GD vs. Subgradient Descent), measuring generalization gap, runtime, and loss variance.

---

## 📁 Repository Structure

```text
SVM/
├── svm_app.py                            # Full interactive Streamlit suite (Main App)
├── fsvm_app.py                           # Enhanced Streamlit app with custom colormaps & documentation
├── svm2_app.py                           # Alternative Streamlit implementation interface
├── hingeloss_app.py                      # Dedicated visualization app for Hinge Loss optimization
├── rbd_app.py                            # Radial Basis Function (RBF) focused dashboard
├── data_banknote_authentication.csv      # Sample Banknote Authentication benchmark dataset
├── project_convex_(2)[1].ipynb           # Jupyter Notebook with derivation & experiments
├── project_convex_kernel_analysis (1).ipynb # Comprehensive Kernel SVM analytical notebook
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the required libraries:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn cvxopt pillow openpyxl
```

### Running the Web Application

To launch the interactive SVM Analyzer dashboard, execute:

```bash
streamlit run svm_app.py
```
*(Or run `streamlit run fsvm_app.py` for the enhanced dashboard version.)*

---

## 🔬 Usage Guide

1. **Upload Dataset**: In the left sidebar, upload a `.csv` or `.xlsx` file (or leave empty to use the default synthetic classification dataset). Select the target column.
2. **Visualize Features**: Explore feature relationships and class separations in Tab 2.
3. **Train Linear SVMs**:
   - Navigate to **Linear SVMs**.
   - Train a **Hard Margin SVM** or adjust the Regularization parameter $C$ and learning rate to train a **Soft Margin SVM** via QP, GD, or Subgradient Descent.
4. **Train Kernel SVMs**:
   - Navigate to **Kernel SVMs** to experiment with non-linear RBF or Polynomial kernels, tuning $C$ and $\gamma$.
5. **Compare Optimization Methods**:
   - Use the **Model Comparison** tab to analyze convergence speed, final loss stability, and test accuracy across different optimization algorithms.
