import numpy as np
import scipy as sp
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import label_binarize


def get_metrics(y_true, y_pred):
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse =  np.sqrt(np.mean(np.power(errors, 2)))
    maxer = np.max(np.abs(errors))
    r2, pval = np.power(sp.stats.pearsonr(y_true.flatten(), y_pred.flatten()), 2)
    if np.isnan(r2):
        print('R2 is nan')
    if np.isinf(r2):
        print('R2 is inf')
    return([mae, rmse, maxer, r2, pval])


def get_cls_metrics(y_true, y_pred, num_cls, digits=6):
    if not any(y_true):
        return None

    num_cls = max(len(set(y_true)), len(set(y_pred)))

    if num_cls > 2:
        y_true_bin = label_binarize(y_true, classes=list(range(num_cls)))
        y_pred_bin = label_binarize(y_pred, classes=list(range(num_cls)))

        try:
            auroc_ovr = roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr')
        except:
            auroc_ovr = 0.0

        try:
            auroc_ovo = roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovo')
        except:
            auroc_ovo = 0.0

        return y_pred, confusion_matrix(y_true, y_pred), auroc_ovr, auroc_ovo,\
                    classification_report(y_true, y_pred, digits=digits), matthews_corrcoef(y_true, y_pred)

    try:
        auroc = roc_auc_score(y_true, y_pred)
    except:
        auroc = 0.0

    return y_pred, confusion_matrix(y_true, y_pred), auroc,\
                    classification_report(y_true, y_pred, digits=digits), matthews_corrcoef(y_true, y_pred)