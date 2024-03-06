import torchmetrics


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


def get_metrics_cls_pt(y_true, y_pred):
    num_classes = y_pred.shape[-1]
    return {
        'AUROC': torchmetrics.functional.auroc(y_pred, y_true, task="binary", num_classes=num_classes),
        'MCC': torchmetrics.functional.matthews_corrcoef(y_pred, y_true, task="binary", num_classes=num_classes)
    }
