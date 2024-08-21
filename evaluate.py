import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# Load saved metrics for loss history
loss_history = np.load('loss_history.npy')
val_loss_history = np.load('val_loss_history.npy')


# Plot Training and Validation Loss Curves
def plot_loss_curves():
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig('loss_curve.png')
    plt.show()


# Function to calculate and plot the ROC curve
def plot_roc_curve(y_true, y_pred):
    # Flatten and convert data types for compatibility
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(float)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()


# Function to calculate and plot the Precision-Recall curve
def plot_precision_recall_curve(y_true, y_pred):
    # Flatten and convert data types for compatibility
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(float)

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    plt.show()


# Function to calculate and plot the Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, threshold=0.5):
    # Flatten and convert data types, then binarize predictions
    y_true = y_true.ravel().astype(int)
    y_pred_binary = (y_pred.ravel() > threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)

    # Plot confusion matrix
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == "__main__":
    # Plot the loss curves for training and validation
    plot_loss_curves()

    # Load ground truth and prediction arrays
    y_true = np.load('y_true.npy')
    y_pred = np.load('y_pred.npy')

    # Plot ROC curve
    plot_roc_curve(y_true, y_pred)

    # Plot Precision-Recall curve
    plot_precision_recall_curve(y_true, y_pred)

    # Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, threshold=0.5)
