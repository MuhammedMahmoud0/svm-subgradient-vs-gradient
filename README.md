# SVM from Scratch: Gradient Descent vs Subgradient Descent

This project implements a Support Vector Machine (SVM) classifier from scratch using Hinge Loss on the [Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication). It compares the performance of **Gradient Descent** and **Subgradient Descent** optimization techniques.

---

## 📁 Project Structure
```
.
├── data_banknote_authentication.csv    # Dataset
├── main.py                             # Entry point for training and evaluation
├── requirements.txt                    # Required Python packages
└── README.md                           # Project documentation
```
---

## 📊 Features

- SVM implementation from scratch (no scikit-learn)
- Hinge Loss function (non-differentiable)
- Gradient Descent and Subgradient Descent optimizers
- Tracks and plots:
  - Loss per epoch
  - Training and validation accuracy
  - Confusion matrix
- Evaluation on convergence, accuracy, and generalization

---

## 📈 Visualizations

- Loss Curve (per epoch)
- Accuracy Curve (training & validation)
- Confusion Matrix

---

## 🧪 Evaluation Metrics

- Training Accuracy
- Validation Accuracy
- Convergence Speed
- Loss Stability
- Generalization to unseen data

---

## 📦 Installation

```bash
git https://github.com/MuhammedMahmoud0/svm-subgradient-vs-gradient.git
cd svm-subgradiant-vs-gradiant
pip install -r requirements.txt
