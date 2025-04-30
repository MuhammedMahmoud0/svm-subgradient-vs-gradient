import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Read CSV
data = pd.read_csv("data_banknote_authentication.csv")

# Correct slicing
X = data.iloc[:, :-1].values  # Take all columns except the last
y = data.iloc[:, -1].values  # Take only the last column

# Convert labels from {0, 1} to {-1, 1}
y = 2 * y - 1


# Define manual train-test split function
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_point = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[indices[:split_point]], X[indices[split_point:]]
    y_train, y_test = y[indices[:split_point]], y[indices[split_point:]]
    return X_train, X_test, y_train, y_test


# Define standardize function
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - mean) / std
    return X_std, mean, std


# Manual split: train (60%), validation (20%), test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split_custom(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split_custom(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

# Standardize manually
X_train, mean, std = standardize(X_train)
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# Hinge loss function
def hinge_loss(w, b, X, y, reg=0.01):
    margins = 1 - y * (np.dot(X, w) + b)
    return np.mean(np.maximum(0, margins)) + reg * np.dot(w, w)


# Logistic loss function
def logistic_loss(w, b, X, y, reg=0.01):
    scores = y * (np.dot(X, w) + b)
    return np.mean(np.log(1 + np.exp(-scores))) + reg * np.dot(w, w)


# Logistic loss gradient
def logistic_gradient(w, b, X, y, reg=0.01):
    n_samples = X.shape[0]
    scores = y * (np.dot(X, w) + b)
    probs = 1 / (1 + np.exp(scores))  # sigmoid(-scores)
    dw = -np.dot((probs * y), X) / n_samples + 2 * reg * w
    db = -np.mean(probs * y)
    return dw, db


# Training function
# def train_svm(
#     X, y, X_val, y_val, method="gradient", lr=0.01, epochs=X_train.shape[0], reg=0.01
# ):
#     n_features = X.shape[1]
#     w = np.zeros(n_features)
#     b = 0
#     loss_hist, acc_hist = [], []
#     val_acc_hist, val_loss_hist = [], []
#
#     for _ in range(epochs):
#         #todo stop condition
#         # --- Compute gradient ---
#         if method == "gradient":
#             dw, db = logistic_gradient(w, b, X, y, reg)
#             train_loss = logistic_loss(w, b, X, y, reg)
#             val_loss = logistic_loss(w, b, X_val, y_val, reg)
#         elif method == "subgradient":
#             margins = 1 - y * (np.dot(X, w) + b)
#             grad_contrib = (-y[:, None] * X) * (margins[:, None] > 0)
#             dw = np.mean(grad_contrib, axis=0) + 2 * reg * w
#             db = np.mean(-y * (margins > 0))
#             train_loss = hinge_loss(w, b, X, y, reg)
#             val_loss = hinge_loss(w, b, X_val, y_val, reg)
#         else:
#             raise ValueError("Unknown method")
#
#         # --- Update weights ---
#         w -= lr * dw
#         b -= lr * db
#
#         # --- Record metrics ---
#         y_pred_train = np.sign(np.dot(X, w) + b)
#         y_pred_val = np.sign(np.dot(X_val, w) + b)
#         train_acc = np.mean(y_pred_train == y)
#         val_acc = np.mean(y_pred_val == y_val)
#
#         loss_hist.append(train_loss)
#         acc_hist.append(train_acc)
#         val_loss_hist.append(val_loss)
#         val_acc_hist.append(val_acc)
#
#     return w, b, loss_hist, acc_hist, val_loss_hist, val_acc_hist


def train_svm(
    X, y, X_val, y_val, method="gradient", lr=0.01, epochs=1000, reg=0.01, tol=1e-4
):
    """
    Train a linear SVM (hinge or logistic) with early stopping based on loss convergence.

    Parameters:
    - X, y: training data and labels
    - X_val, y_val: validation data and labels
    - method: 'gradient' (logistic gradient descent) or 'subgradient' (hinge subgradient descent)
    - lr: learning rate
    - epochs: maximum number of epochs
    - reg: L2 regularization strength
    - tol: tolerance threshold for early stopping (difference in training loss)

    Returns:
    - w, b: learned weights and bias
    - loss_hist, acc_hist: training loss and accuracy history
    - val_loss_hist, val_acc_hist: validation loss and accuracy history
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0
    loss_hist, acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # Compute gradient and losses
        if method == "gradient":
            dw, db = logistic_gradient(w, b, X, y, reg)
            train_loss = logistic_loss(w, b, X, y, reg)
            val_loss = logistic_loss(w, b, X_val, y_val, reg)
        elif method == "subgradient":
            margins = 1 - y * (X.dot(w) + b)
            mask = margins > 0
            grad_contrib = (-y[:, None] * X) * mask[:, None]
            dw = np.mean(grad_contrib, axis=0) + 2 * reg * w
            db = np.mean(-y * mask)
            train_loss = hinge_loss(w, b, X, y, reg)
            val_loss = hinge_loss(w, b, X_val, y_val, reg)
        else:
            raise ValueError("Unknown method")

        # Update parameters
        w -= lr * dw
        b -= lr * db

        # Record metrics
        loss_hist.append(train_loss)
        acc = np.mean(np.sign(X.dot(w) + b) == y)
        acc_hist.append(acc)
        val_loss_hist.append(val_loss)
        val_acc = np.mean(np.sign(X_val.dot(w) + b) == y_val)
        val_acc_hist.append(val_acc)

        # Early stopping: check loss convergence
        if epoch > 0 and abs(val_loss_hist[-1] - val_loss_hist[-2]) <= tol:
            break

    return w, b, loss_hist, acc_hist, val_loss_hist, val_acc_hist


# Train using both methods
w_grad, b_grad, loss_grad, acc_grad, val_loss_grad, val_acc_grad = train_svm(
    X_train, y_train, X_val, y_val, method="gradient"
)
w_sub, b_sub, loss_sub, acc_sub, val_loss_sub, val_acc_sub = train_svm(
    X_train, y_train, X_val, y_val, method="subgradient"
)

# Test accuracy
test_pred_grad = np.sign(np.dot(X_test, w_grad) + b_grad)
test_pred_sub = np.sign(np.dot(X_test, w_sub) + b_sub)
test_acc_grad = np.mean(test_pred_grad == y_test)
test_acc_sub = np.mean(test_pred_sub == y_test)

# Plot Loss Curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(
    loss_grad,
    label="Gradient Descent",
    linestyle="-",
    color="blue",
    marker="o",
    markersize=3,
)
plt.plot(
    loss_sub,
    label="Subgradient Descent",
    linestyle="--",
    color="red",
    marker="x",
    markersize=3,
)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Hinge Loss")
plt.legend()
plt.grid(True)

# Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(
    val_acc_grad,
    label="Gradient Descent",
    linestyle="-",
    color="blue",
    marker="o",
    markersize=3,
)
plt.plot(
    val_acc_sub,
    label="Subgradient Descent",
    linestyle="--",
    color="red",
    marker="x",
    markersize=3,
)
plt.title(" Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Confusion Matrices
def plot_confusion(y_true, y_pred, title, fig_rows, fig_columns, index):
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    plt.subplot(fig_rows, fig_columns, index)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Positive", "Negative"],
        yticklabels=["Positive", "Negative"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")


plt.figure(figsize=(12, 5))
plot_confusion(y_test, test_pred_grad, "Confusion Matrix (Gradient Descent)", 1, 2, 1)
plot_confusion(y_test, test_pred_sub, "Confusion Matrix (Subgradient Descent)", 1, 2, 2)
plt.tight_layout()
plt.show()

# Print comparison metrics
# {
#     "Gradient Descent Test Accuracy": test_acc_grad,
#     "Subgradient Descent Test Accuracy": test_acc_sub,
#     "Gradient Descent Final Loss": loss_grad[-1],
#     "Subgradient Descent Final Loss": loss_sub[-1],
# }
{
    "Gradient Descent Test Accuracy": float(round(test_acc_grad*100, 2)),
    "Subgradient Descent Test Accuracy": float(round(test_acc_sub*100, 2)),
    "Gradient Descent Final Loss": float(round(loss_grad[-1], 3)),
    "Subgradient Descent Final Loss": float(round(loss_sub[-1], 3))
}


len(val_loss_grad), len(val_loss_sub)

# Confusion matrices for both methods
cm_grad = confusion_matrix(y_test, test_pred_grad, labels=[1, -1])
cm_sub = confusion_matrix(y_test, test_pred_sub, labels=[1, -1])

# Extract values from the confusion matrix for Gradient Descent
TP_grad = cm_grad[0, 0]  # True Positive (Gradient Descent)
TN_grad = cm_grad[1, 1]  # True Negative (Gradient Descent)
FP_grad = cm_grad[0, 1]  # False Positive (Gradient Descent)
FN_grad = cm_grad[1, 0]  # False Negative (Gradient Descent)

# Extract values from the confusion matrix for Subgradient Descent
TP_sub = cm_sub[0, 0]  # True Positive (Subgradient Descent)
TN_sub = cm_sub[1, 1]  # True Negative (Subgradient Descent)
FP_sub = cm_sub[0, 1]  # False Positive (Subgradient Descent)
FN_sub = cm_sub[1, 0]  # False Negative (Subgradient Descent)

# Compute Precision and Recall for Gradient Descent
precision_grad = TP_grad / (TP_grad + FP_grad) if (TP_grad + FP_grad) != 0 else 0
recall_grad = TP_grad / (TP_grad + FN_grad) if (TP_grad + FN_grad) != 0 else 0

# Compute Precision and Recall for Subgradient Descent
precision_sub = TP_sub / (TP_sub + FP_sub) if (TP_sub + FP_sub) != 0 else 0
recall_sub = TP_sub / (TP_sub + FN_sub) if (TP_sub + FN_sub) != 0 else 0

# Print the results
print(f"Gradient Descent - Precision: {precision_grad:.2f}, Recall: {recall_grad:.2f}")
print(f"Subgradient Descent - Precision: {precision_sub:.2f}, Recall: {recall_sub:.2f}")
