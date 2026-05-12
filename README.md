# 📊 MATH 3410: Probability & Statistics I
**Weber State University** · Interactive Learning Apps

> A cookiecutter-style collection of interactive Streamlit apps and web tools for exploring core concepts in probability and statistics — built for classroom use and self-study.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Apps-red.svg)](https://streamlit.io)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://jishanahmed2019.github.io/WSUMath3410/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📁 Project Structure

```
WSUMath3410/
├── README.md                     ← You are here
├── requirements.txt              ← Python dependencies
├── packages.txt                  ← System packages (for Streamlit Cloud)
├── runtime.txt                   ← Python runtime version
│
├── apps/                         ← All Streamlit Python apps
│   ├── probability/              ← Foundations of probability
│   │   ├── Bayes.py
│   │   ├── MontyHall.py
│   │   ├── coin.py
│   │   └── Set.py
│   ├── distributions/            ← Distributions & density functions
│   │   ├── PDF_CDF.py
│   │   ├── CLT_Final.py
│   │   ├── PoissonMLE.py
│   │   └── KS.py
│   ├── inference/                ← Hypothesis testing & confidence intervals
│   │   ├── CI.py
│   │   ├── HypothesisTestDeploy.py
│   │   └── t_test_app.py
│   ├── regression/               ← Regression & forecasting
│   │   ├── MultipleLinearRegression.py
│   │   ├── MultipleLinearRegressionTest.py
│   │   ├── Corr.py
│   │   └── StockPrice.py
│   ├── ml/                       ← Machine learning concepts
│   │   ├── KNN.py
│   │   ├── Clustering.py
│   │   └── mixedICAFinal.py
│   └── optimization/             ← Optimization & linear algebra
│       ├── Newton.py
│       ├── localMax.py
│       └── SVD.py
│
├── static/                       ← Static HTML apps & media
│   ├── html/
│   │   ├── CountingApps.html
│   │   ├── DiceRoll.html
│   │   ├── DiceRollSmall.html
│   │   └── visited_cities_map.html
│   └── images/
│       └── math_horiz.png
│
├── notebooks/                    ← Jupyter notebooks for exploration
│   └── explorations/
│
├── src/                          ← Shared utilities & helper modules
│   └── supplementary.py
│
└── docs/                         ← App screenshots & course documentation
```

---

## 🗂️ Apps by Topic

### 🎲 Probability Foundations — `apps/probability/`

| App | Description | Run |
|-----|-------------|-----|
| `Bayes.py` | Bayes' Theorem — interactive calculator and visualizer | `streamlit run apps/probability/Bayes.py` |
| `MontyHall.py` | Monty Hall problem — run thousands of trials and see convergence | `streamlit run apps/probability/MontyHall.py` |
| `coin.py` | Coin toss simulator — Law of Large Numbers in action | `streamlit run apps/probability/coin.py` |
| `Set.py` | Set operations with Venn diagrams — union, intersection, complement | `streamlit run apps/probability/Set.py` |

### 📈 Distributions & Density — `apps/distributions/`

| App | Description | Run |
|-----|-------------|-----|
| `PDF_CDF.py` | Visualize PDFs and CDFs for common distributions | `streamlit run apps/distributions/PDF_CDF.py` |
| `CLT_Final.py` | Central Limit Theorem — watch distributions normalize with sample size | `streamlit run apps/distributions/CLT_Final.py` |
| `PoissonMLE.py` | Poisson distribution with Maximum Likelihood Estimation | `streamlit run apps/distributions/PoissonMLE.py` |
| `KS.py` | Kolmogorov–Smirnov test for goodness of fit | `streamlit run apps/distributions/KS.py` |

### 🔬 Statistical Inference — `apps/inference/`

| App | Description | Run |
|-----|-------------|-----|
| `CI.py` | Confidence interval explorer — adjust confidence level and sample size | `streamlit run apps/inference/CI.py` |
| `HypothesisTestDeploy.py` | Hypothesis testing using the T-test model | `streamlit run apps/inference/HypothesisTestDeploy.py` |
| `t_test_app.py` | One-sample and two-sample T-test interactive app | `streamlit run apps/inference/t_test_app.py` |

### 📉 Regression & Prediction — `apps/regression/`

| App | Description | Run |
|-----|-------------|-----|
| `MultipleLinearRegression.py` | Multiple linear regression with interactive input | `streamlit run apps/regression/MultipleLinearRegression.py` |
| `MultipleLinearRegressionTest.py` | Extended MLR test suite | `streamlit run apps/regression/MultipleLinearRegressionTest.py` |
| `Corr.py` | Correlation visualization | `streamlit run apps/regression/Corr.py` |
| `StockPrice.py` | Stock price forecasting using regression models | `streamlit run apps/regression/StockPrice.py` |

### 🤖 Machine Learning — `apps/ml/`

| App | Description | Run |
|-----|-------------|-----|
| `KNN.py` | K-Nearest Neighbors classifier — interactive decision boundaries | `streamlit run apps/ml/KNN.py` |
| `Clustering.py` | Image segmentation using clustering algorithms | `streamlit run apps/ml/Clustering.py` |
| `mixedICAFinal.py` | Independent Component Analysis (ICA) demo | `streamlit run apps/ml/mixedICAFinal.py` |

### 🔧 Optimization & Linear Algebra — `apps/optimization/`

| App | Description | Run |
|-----|-------------|-----|
| `Newton.py` | Newton's method for root finding | `streamlit run apps/optimization/Newton.py` |
| `localMax.py` | Local maxima finder with visualization | `streamlit run apps/optimization/localMax.py` |
| `SVD.py` | Singular Value Decomposition visualization | `streamlit run apps/optimization/SVD.py` |

### 🌐 Static HTML Apps — `static/html/`

> Open directly in a browser — no server needed.

| App | Description |
|-----|-------------|
| `CountingApps.html` | Counting principles: permutations and combinations |
| `DiceRoll.html` | Dice roll simulator with frequency charts |
| `DiceRollSmall.html` | Lightweight dice roll demo |
| `visited_cities_map.html` | Interactive cities map |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/JishanAhmed2019/WSUMath3410.git
cd WSUMath3410
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run any app

```bash
# Example — launch the Bayes' Theorem app
streamlit run apps/probability/Bayes.py
```

### 4. Open an HTML app

```bash
open static/html/DiceRoll.html
# or just double-click the file in your file explorer
```

---

## ☁️ Deployment

Apps are deployed via **Streamlit Community Cloud** and **GitHub Pages**.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Interactive Python web apps |
| [NumPy / SciPy](https://scipy.org) | Statistical computation |
| [Matplotlib / Plotly](https://plotly.com) | Data visualization |
| [scikit-learn](https://scikit-learn.org) | ML algorithms (KNN, clustering) |
| GitHub Pages | Static HTML app hosting |

---

## 👤 Author

**Jishan Ahmed** — [JishanAhmed2019](https://github.com/JishanAhmed2019)
Weber State University · Department of Mathematics

---

*Built for MATH 3410. Fork freely for your own courses.*
