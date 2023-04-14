import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score, f1_score,
    precision_recall_fscore_support, mean_squared_error, mean_absolute_error,
    explained_variance_score)
from torch.nn.functional import one_hot


def make_eval_metrics_classification(
        lightning_module, y, logits, phase='validation'):
    y_pred = logits.argmax(axis=1)
    y_ohot = one_hot(y, num_classes=logits.shape[1])

    # Compute accuracy and log
    accuracy = accuracy_score(y, y_pred)
    lightning_module.log("accuracy", accuracy)

    # Compute F1 macro and micro and log
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_micro = f1_score(y, y_pred, average="micro")
    lightning_module.log(f"{phase}_f1_macro", f1_macro)
    lightning_module.log(f"{phase}_f1_micro", f1_micro)

    # Compute precision and recall and log
    precision, recall, _, _ = precision_recall_fscore_support(
        y, y_pred, average="weighted")
    lightning_module.log(f"{phase}_precision", precision)
    lightning_module.log(f"{phase}_recall", recall)

    # Plot confusion matrix
    confusion_mat = confusion_matrix(
        y, y_pred, labels=range(logits.shape[1]))
    plt.figure()
    plt.matshow(confusion_mat)
    plt.title('Confusion Matrix')
    # Save heamap as an artifact
    plt.savefig('confusion_matrix.png', format='png')
    lightning_module.logger.experiment.log_artifact(
        local_path='confusion_matrix.png',
        artifact_path=f"{phase}/confusion_matrix",
        run_id=lightning_module.logger.run_id)
    os.remove('confusion_matrix.png')
    plt.close()

    # Compute ROC curve and AUC
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(lightning_module.num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_ohot[:, i], logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        lightning_module.log(
            f"{phase}_ROC_AUC_class_{i}", roc_auc[i]
        )

        # Plot ROC curve
        plt.figure()
        plt.plot(
            fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Class {i}')
        plt.legend(loc="lower right")
        plt.grid()
        # Save ROC curve plot as an artifact
        plt.savefig(f'class_{i}.png', format='png')
        lightning_module.logger.experiment.log_artifact(
            local_path=f'class_{i}.png',
            artifact_path=f"{phase}/roc_curve",
            run_id=lightning_module.logger.run_id)
        os.remove(f'class_{i}.png')
        plt.close()


def make_eval_metrics_regression(
        lightning_module, y, y_pred, phase='validation'):
    # Compute mean squared error and log
    rmse = mean_squared_error(y, y_pred) ** (1/2)
    lightning_module.log(f"{phase}_mse", rmse)

    # Compute mean absolute error and log
    mae = mean_absolute_error(y, y_pred)
    lightning_module.log(f"{phase}_mae", mae)

    # Compute explained variance and log
    exp_var = explained_variance_score(y, y_pred)
    lightning_module.log(f"{phase}_explained_variance", exp_var)

    # Create a scatter plot of actual vs predicted values
    plt.figure()
    plt.scatter(y, y_pred, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid()

    # Save scatter plot as an artifact
    plt.savefig(f'{phase}_scatter_plot.png', format='png')
    lightning_module.logger.experiment.log_artifact(
        local_path=f'{phase}_scatter_plot.png',
        artifact_path=f"{phase}/scatter_plot",
        run_id=lightning_module.logger.run_id)
    os.remove(f'{phase}_scatter_plot.png')
    plt.close()

    # Compute residuals
    residuals = y - y_pred

    # Create a histogram of residuals
    plt.figure()
    plt.hist(residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.grid()

    # Save histogram as an artifact
    plt.savefig(f'{phase}_histogram_residuals.png', format='png')
    lightning_module.logger.experiment.log_artifact(
        local_path=f'{phase}_histogram_residuals.png',
        artifact_path=f"{phase}/histogram_residuals",
        run_id=lightning_module.logger.run_id)
    os.remove(f'{phase}_histogram_residuals.png')
    plt.close()