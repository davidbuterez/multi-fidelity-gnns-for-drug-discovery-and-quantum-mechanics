import numpy as np
import torchmetrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score, confusion_matrix, roc_auc_score, classification_report, matthews_corrcoef


def get_metrics_pt(y_true, y_pred):
    # Most metrics are commented as they can take a long time to compute

    # r = torchmetrics.functional.pearson_corrcoef(y_pred, y_true)

    return {
        # 'ConcordanceCorrCoef': torchmetrics.functional.concordance_corrcoef(y_pred, y_true),
        # 'ExplainedVariance': torchmetrics.functional.explained_variance(y_pred, y_true),
        # 'KendallRankCorrCoef': torchmetrics.functional.kendall_rank_corrcoef(y_pred, y_true),
        'MAE': torchmetrics.functional.mean_absolute_error(y_pred, y_true),
        # 'MSE': torchmetrics.functional.mean_squared_error(y_pred, y_true),
        'RMSE': torchmetrics.functional.mean_squared_error(y_pred, y_true, squared=False),
        # 'PearsonCorrCoef': r,
        # 'PearsonCorrCoefSquared': r ** 2,
        'R2': torchmetrics.functional.r2_score(y_pred, y_true),
        # 'SpearmanCorrCoef': torchmetrics.functional.spearman_corrcoef(y_pred, y_true),
        # 'SMAPE': torchmetrics.functional.symmetric_mean_absolute_percentage_error(y_pred, y_true)
    }


def get_cls_metrics(y_true, y_pred, digits=6):
    conf_matrix = confusion_matrix(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, digits=digits)
    mcc = matthews_corrcoef(y_true, y_pred)

    return conf_matrix, auroc, cls_report, mcc


def get_metrics_qm7(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    maxerr = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return [mae, rmse, maxerr, r2]