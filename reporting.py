import numpy as np
import torchmetrics
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef, r2_score


def get_metrics(y_true, y_pred):
    errors = y_true.flatten() - y_pred.flatten()
    mae = np.mean(np.abs(errors))
    rmse =  np.sqrt(np.mean(np.power(errors, 2)))
    maxer = np.max(np.abs(errors))
    r2 = r2_score(y_true.flatten(), y_pred.flatten())

    return([mae, rmse, maxer, r2])


def get_metrics_pt(y_true, y_pred):
    r = torchmetrics.functional.pearson_corrcoef(y_pred, y_true)

    return {
        'ConcordanceCorrCoef': torchmetrics.functional.concordance_corrcoef(y_pred, y_true),
        'ExplainedVariance': torchmetrics.functional.explained_variance(y_pred, y_true),
        'KendallRankCorrCoef': torchmetrics.functional.kendall_rank_corrcoef(y_pred, y_true),
        'MAE': torchmetrics.functional.mean_absolute_error(y_pred, y_true),
        'MSE': torchmetrics.functional.mean_squared_error(y_pred, y_true),
        'RMSE': torchmetrics.functional.mean_squared_error(y_pred, y_true, squared=False),
        'PearsonCorrCoef': r,
        'PearsonCorrCoefSquared': r ** 2,
        'R2': torchmetrics.functional.r2_score(y_pred, y_true),
        'SpearmanCorrCoef': torchmetrics.functional.spearman_corrcoef(y_pred, y_true),
        'SMAPE': torchmetrics.functional.symmetric_mean_absolute_percentage_error(y_pred, y_true)
    }


def get_cls_metrics(y_true, y_pred, digits=6):
    conf_matrix = confusion_matrix(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, digits=digits)
    mcc = matthews_corrcoef(y_true, y_pred)

    return conf_matrix, auroc, cls_report, mcc
