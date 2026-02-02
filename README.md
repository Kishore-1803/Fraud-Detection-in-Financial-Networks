# 🔍 Fraud Detection in Financial Networks

<p align="center">
  <strong>An AI / Graph-Based Framework for Identifying Anomalous and Fraudulent Activity in Financial Transaction Networks.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ML-Graph%20Learning-orange.svg"/>
  <img src="https://img.shields.io/badge/Tech-Stack-Python%2C%20NetworkX%2C%20scikit-learn-blue.svg"/>
  <img src="https://img.shields.io/badge/Status-Completed-success.svg"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg"/>
</p>

---

## 📌 Project Overview

Fraud detection in financial systems is a critical challenge due to the sophisticated and evolving nature of fraudulent behaviors.

This project applies **network analysis**, **feature engineering**, **machine learning models**, and **node embedding methods** to detect anomalies and potential fraud in simulated or real financial transaction networks.

The focus is on capturing both **structural patterns** and **transaction behaviors** that distinguish normal users from potentially fraudulent entities.

---

## 🚀 Key Features

- 🕸️ **Network Construction**
  - Transforms transaction datasets into graph representations  
  - Nodes represent accounts or entities
  - Edges represent financial interactions

- 💡 **Feature Engineering**
  - Degree statistics
  - Centrality measures
  - Neighborhood aggregation
  - Temporal transaction patterns

- 📊 **Modeling & Algorithms**
  - Baseline models: Logistic Regression, Random Forest, XGBoost
  - Graph-based embeddings: Node2Vec / DeepWalk / GraphSAGE (optional)
  - Comparison and performance analysis across methods

- 📈 **Evaluation Metrics**
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC for threshold performance
  - Confusion matrices and PR curves

- 🔍 **Visualization**
  - Network graphs showing clusters and anomalies
  - Embedding visualizations (e.g., t-SNE / UMAP)
  - Feature importance plots

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Code | Python |
| Data | pandas, NumPy |
| Graph Analytics | NetworkX, node embedding libraries |
| Models | scikit-learn, XGBoost |
| Visualization | matplotlib, seaborn, UMAP, t-SNE |
| Version Control | Git & GitHub |

---

## ⚙️ Getting Started

### 🔹 Prerequisites

Install Python **3.8+** and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

---

### 🧰 Installation

```bash
git clone https://github.com/Kishore-1803/Fraud-Detection-in-Financial-Networks.git
cd Fraud-Detection-in-Financial-Networks
pip install -r requirements.txt
```

---

## 🧠 Workflow & Usage

### 1️⃣ Load and Inspect Data

Use the notebooks or `data_processing.py` to load and explore raw financial transaction data.

### 2️⃣ Construct a Graph

Run graph builder utilities:

```python
from src.graph_builder import build_graph
G = build_graph(transaction_df)
```

Example output: a NetworkX graph with accounts as nodes and transactions as edges.

---

### 3️⃣ Feature Engineering

```python
from src.features import extract_features
features_df = extract_features(G)
```

Includes statistics like:

* Node degree
* Centrality measures
* Neighborhood transactions

---

### 4️⃣ Model Training

```python
from src.models import train_model
model = train_model(features_df, labels)
```

Evaluates:

* Random Forest
* Logistic Regression
* XGBoost
* Graph Embedding + Classifier

---

### 5️⃣ Evaluation & Visualization

* ROC curves
* PR curves
* Embedding plots
* Anomaly detection highlight

Use notebooks for rich explorations.

---

## 📊 Evaluation Metrics

| Metric        | Description                     |
| ------------- | ------------------------------- |
| **Accuracy**  | Overall correctness             |
| **Precision** | Correct fraud predictions       |
| **Recall**    | Detection rate of fraud         |
| **F1-Score**  | Balance of precision and recall |
| **ROC-AUC**   | Threshold robustness            |

---

## 🧪 Example Results

Typical outputs include:

* Confusion matrix figures
* Feature importance chart
* Network cluster visualization
* Embedding scatter plots (t-SNE / UMAP)

---

## 🧠 Why It Matters

* Financial fraud costs billions annually
* Graph-based methods capture relational structures that traditional models miss
* Visualization provides explainable insights, not black-box predictions

---

## 🔍 Future Enhancements

* Integrate temporal network analysis
* Deploy real-time detection pipelines
* Replace simulated data with production banking datasets
* Integrate GNN models: GCN, GraphSAGE
* Build dashboard for real-time insights

---

## 📄 License

This project is licensed under the **MIT License — see LICENSE for details.**

---

⭐ *If you found this repository helpful, please consider starring it!* ⭐
