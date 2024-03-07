import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle

# load model
best_cb_clf = joblib.load('best_cb_clf.joblib')
best_log_reg = joblib.load('best_log_reg.joblib')
best_rf_clf = joblib.load('best_rf_clf.joblib')

# load data
df = pd.read_csv('filled_chosen.csv', engine='python')

X = df.drop('aki', axis=1)
y = df['aki']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# prediction
cb_clf_pred = best_cb_clf.predict(X_test)
log_reg_pred = best_log_reg.predict(X_test)
rf_clf_pred = best_rf_clf.predict(X_test)

# evaluation
cb_clf_acc = accuracy_score(y_test, cb_clf_pred)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
rf_clf_acc = accuracy_score(y_test, rf_clf_pred)

# auc
cb_clf_proba = best_cb_clf.predict_proba(X_test)
cb_clf_auc = roc_auc_score(y_test, cb_clf_proba, multi_class='ovr', average='macro')

log_reg_proba = best_log_reg.predict_proba(X_test)
log_reg_auc = roc_auc_score(y_test, log_reg_proba, multi_class='ovr', average='macro')

rf_clf_proba = best_rf_clf.predict_proba(X_test)
rf_clf_auc = roc_auc_score(y_test, rf_clf_proba, multi_class='ovr', average='macro')


# performance
print(f"Gradient Boosting Accuracy: {cb_clf_acc}, AUC: {cb_clf_auc}")
print(f"Logistic Regression Accuracy: {log_reg_acc}, AUC: {log_reg_auc}")
print(f"Random Forest Accuracy: {rf_clf_acc}, AUC: {rf_clf_auc}")

# polt the result
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

colors = cycle(['blue', 'red', 'green'])

# Confusion matrix
def plot_confusion_matrix_for_model(model, X_test, y_test, title, ax=None):
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                 cmap=plt.cm.Blues,
                                                 normalize=None, ax=ax)
    disp.ax_.set_title(title)
    disp.im_.colorbar.remove()
    plt.tight_layout()

# 2x3 plot layout
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plotting Confusion Matrices
plot_confusion_matrix_for_model(best_cb_clf, X_test, y_test, 'Gradient Boosting', ax=axs[0, 0])
plot_confusion_matrix_for_model(best_log_reg, X_test, y_test, 'Logistic Regression', ax=axs[0, 1])
plot_confusion_matrix_for_model(best_rf_clf, X_test, y_test, 'Random Forest Classifier', ax=axs[0, 2])

def plot_multiclass_roc(model, X_test, y_test, n_classes, title, ax=None):
    y_score = model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label='Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")

# Plotting ROC Curves
for i, model in enumerate([best_cb_clf, best_log_reg, best_rf_clf]):
    plot_multiclass_roc(model, X_test, y_test, n_classes=len(np.unique(y_test)),
                        title=['Gradient Boosting ROC Curve', 'Logistic Regression ROC Curve', 'Random Forest ROC Curve'][i],
                        ax=axs[1, i])

plt.tight_layout()
plt.show()
