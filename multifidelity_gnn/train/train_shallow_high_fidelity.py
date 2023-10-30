from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

import pandas as pd
import numpy as np
import argparse
import os
import torch

from joblib import dump
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

from ..src.reporting import get_metrics_pt, get_cls_metrics

# import warnings
# warnings.filterwarnings('ignore')


def remove_smiles_stereo(s):
    mol = Chem.MolFromSmiles(s)
    Chem.rdmolops.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(mol)


def get_metrics(y_pred, y_true, task_type):
    if task_type == "regression":
        y_true = torch.from_numpy(y_true)
        y_pred = torch.from_numpy(y_pred)
        metrics = get_metrics_pt(y_true, y_pred)
        metrics = {k: v.item() for k, v in metrics.items()}
    else:
        metrics = get_cls_metrics(y_true, y_pred, 2)

    return metrics


def save_joblib(obj, name, save_path):
    Path(save_path).mkdir(exist_ok=True, parents=True)
    dump(obj, os.path.join(save_path, f"{name}.joblib"))


def train_model(model, x_train, y_train, x_test, parameters=None, model_type="rf"):
    if model_type == "svm":
        args = {}
    else:
        args = {"n_jobs": -1}

    if parameters is not None:
        estimator = model(**args, **parameters)
    else:
        estimator = model(**args)

    estimator.fit(x_train, y_train)
    preds = estimator.predict(x_test)

    return preds, model


def perform_grid_search(
    model,
    grid,
    predefined_split,
    scoring,
    x_train_val,
    y_train_val,
):
    grid_search = GridSearchCV(
        estimator=model(),
        param_grid=grid,
        cv=predefined_split,
        scoring=scoring,
        verbose=0,
    )

    grid_search.fit(x_train_val, y_train_val)

    return grid_search


def get_fps(train_mols, valid_mols, test_mols):
    fps_train = np.array(
        [
            np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
            for mol in train_mols
        ]
    )

    fps_valid = np.array(
        [
            np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
            for mol in valid_mols
        ]
    )

    fps_test = np.array(
        [
            np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
            for mol in test_mols
        ]
    )

    return fps_train, fps_valid, fps_test


def get_pc(train_mols, valid_mols, test_mols):
    all_feats = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_feats)
    print("Using %i PhysChem features " % (len(all_feats)))

    train_descrs = [calc.CalcDescriptors(mol) for mol in train_mols]
    valid_descrs = [calc.CalcDescriptors(mol) for mol in valid_mols]
    test_descrs = [calc.CalcDescriptors(mol) for mol in test_mols]

    train_descrs = np.nan_to_num(np.array(train_descrs))
    test_descrs = np.nan_to_num(np.array(test_descrs))
    valid_descrs = np.nan_to_num(np.array(valid_descrs))

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_descrs)

    X_train = scaler.transform(train_descrs)
    X_valid = scaler.transform(valid_descrs)
    X_test = scaler.transform(test_descrs)

    return X_train, X_valid, X_test


def add_label(
    x_train, x_val, x_test, train_df, val_df, test_df, train_label, eval_label
):
    x_train_lbl = np.concatenate(
        (x_train, np.expand_dims(train_df[train_label].values, axis=1)), axis=1
    )
    x_val_lbl = np.concatenate(
        (x_val, np.expand_dims(val_df[eval_label].values, axis=1)), axis=1
    )
    x_test_lbl = np.concatenate(
        (x_test, np.expand_dims(test_df[eval_label].values, axis=1)), axis=1
    )

    return x_train_lbl, x_val_lbl, x_test_lbl


def add_embeddings(x_train, x_val, x_test, train_df, val_df, test_df, embs_label):
    train_embs = np.array(
        [
            np.array(
                arr.rstrip().lstrip().replace("[", "").replace("]", "").split(","),
                dtype=float,
            )
            for arr in train_df[embs_label].values
        ]
    )

    val_embs = np.array(
        [
            np.array(
                arr.rstrip().lstrip().replace("[", "").replace("]", "").split(","),
                dtype=float,
            )
            for arr in val_df[embs_label].values
        ]
    )

    test_embs = np.array(
        [
            np.array(
                arr.rstrip().lstrip().replace("[", "").replace("]", "").split(","),
                dtype=float,
            )
            for arr in test_df[embs_label].values
        ]
    )

    x_train_embs = np.concatenate((x_train, train_embs), axis=1)
    x_val_embs = np.concatenate((x_val, val_embs), axis=1)
    x_test_embs = np.concatenate((x_test, test_embs), axis=1)

    return x_train_embs, x_val_embs, x_test_embs


def main():
    parser = argparse.ArgumentParser(description="RF/SVR training on DR.")
    parser.add_argument("--data-path")
    parser.add_argument("--smiles-column")
    parser.add_argument("--DR-label")
    parser.add_argument("--train-SD-label")
    parser.add_argument("--eval-SD-label")
    parser.add_argument("--SD-EMBS-label")
    parser.add_argument("--lbl-or-emb")
    parser.add_argument("--fp-or-pc")
    parser.add_argument("--rf-or-svm")
    parser.add_argument("--task-type")
    parser.add_argument("--save-path")
    args = parser.parse_args()
    argsdict = vars(args)

    assert argsdict["task_type"] in ["regression", "classification"]
    assert argsdict["lbl_or_emb"] in ["base", "lbl", "emb"]
    assert argsdict["fp_or_pc"] in ["fp", "pc"]
    assert argsdict["rf_or_svm"] in ["rf", "svm"]

    scoring = (
        "roc_auc"
        if argsdict["task_type"] == "classification"
        else "neg_mean_squared_error"
    )
    if argsdict["rf_or_svm"] == "rf":
        print("Using RF.")
        model = (
            RandomForestClassifier
            if argsdict["task_type"] == "classification"
            else RandomForestRegressor
        )
    elif argsdict["rf_or_svm"] == "svm":
        print("Using SVM.")
        model = SVC if argsdict["task_type"] == "classification" else SVR

    save_path = argsdict["save_path"]
    Path(save_path).mkdir(exist_ok=True, parents=True)

    data_path = argsdict["data_path"]
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_path, "validate.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    train_smiles_no_stereo = [
        remove_smiles_stereo(smiles)
        for smiles in train_df[argsdict["smiles_column"]].values
    ]
    val_smiles_no_stereo = [
        remove_smiles_stereo(smiles)
        for smiles in val_df[argsdict["smiles_column"]].values
    ]
    test_smiles_no_stereo = [
        remove_smiles_stereo(smiles)
        for smiles in test_df[argsdict["smiles_column"]].values
    ]

    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles_no_stereo]
    valid_mols = [Chem.MolFromSmiles(s) for s in val_smiles_no_stereo]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles_no_stereo]

    y_train = train_df[argsdict["DR_label"]].values
    y_val = val_df[argsdict["DR_label"]].values
    y_test = test_df[argsdict["DR_label"]].values

    if argsdict["task_type"] == "regression":
        y_scaler = preprocessing.StandardScaler()
        y_scaler.fit_transform(y_train.reshape(-1, 1))

        y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))
        y_valid_scaled = y_scaler.transform(y_val.reshape(-1, 1))

        y_train_val_scaled = np.concatenate([y_train_scaled, y_valid_scaled]).reshape(
            -1,
        )

        y_train_scaled = y_train_scaled.ravel()
        y_valid_scaled = y_valid_scaled.ravel()

        y_train = y_train_scaled
        y_val = y_valid_scaled

        y_train_val = y_train_val_scaled
    else:
        y_train_val = np.concatenate([y_train, y_val]).reshape(
            -1,
        )

    if argsdict["fp_or_pc"] == "fp":
        print("Using FP as input data type...")
        x_train, x_val, x_test = get_fps(train_mols, valid_mols, test_mols)
    elif argsdict["fp_or_pc"] == "pc":
        print("Using PC as input data type...")
        x_train, x_val, x_test = get_pc(train_mols, valid_mols, test_mols)

    if argsdict["lbl_or_emb"] == "lbl":
        print("Adding SD labels to data...")
        x_train, x_val, x_test = add_label(
            x_train,
            x_val,
            x_test,
            train_df,
            val_df,
            test_df,
            argsdict["train_SD_label"],
            argsdict["eval_SD_label"],
        )
    elif argsdict["lbl_or_emb"] == "emb":
        print("Adding SD embeddings to data...")
        x_train, x_val, x_test = add_embeddings(
            x_train, x_val, x_test, train_df, val_df, test_df, argsdict["SD_EMBS_label"]
        )

    print(f"Data has {x_train.shape[-1]} features!")

    x_train_val = np.concatenate([x_train, x_val])

    valid_fold = [-1] * len(train_mols) + [0] * len(valid_mols)
    ps = PredefinedSplit(valid_fold)

    NAME = f'{argsdict["rf_or_svm"]}+{argsdict["fp_or_pc"]}+{argsdict["lbl_or_emb"]}'

    if argsdict["lbl_or_emb"] == "lbl":
        aug_for_path = argsdict["train_SD_label"] + argsdict["eval_SD_label"]
    elif argsdict["lbl_or_emb"] == "emb":
        aug_for_path = argsdict["SD_EMBS_label"]
    else:
        aug_for_path = ""

    if aug_for_path != "":
        NAME += f"+{aug_for_path}"

    OUT_PATH = os.path.join(argsdict["save_path"], NAME)
    Path(OUT_PATH).mkdir(exist_ok=True, parents=True)

    rf_random_grid = {
        "n_estimators": [50, 100, 150, 200, 250, 300, 500],
        "max_depth": [None, 25, 50, 100],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
    }

    svm_random_grid = {"C": np.logspace(-2, 3, 10), "gamma": np.logspace(-5, 2, 20)}

    grid = rf_random_grid if argsdict["rf_or_svm"] == "rf" else svm_random_grid

    # Grid search
    print("Starting grid search...")
    grid_search_result = perform_grid_search(
        model, grid, ps, scoring, x_train_val, y_train_val
    )
    print("Finished grid search!")
    save_joblib(grid_search_result, "grid_search", OUT_PATH)

    gs_preds, gs_estimator = train_model(
        model,
        x_train,
        y_train,
        x_test,
        parameters=grid_search_result.best_params_,
        model_type=argsdict["rf_or_svm"],
    )
    save_joblib(gs_estimator, "grid_search_estimator", OUT_PATH)

    if argsdict["task_type"] == "regression":
        gs_preds = y_scaler.inverse_transform(gs_preds.reshape(-1, 1)).flatten()

    np.save(os.path.join(OUT_PATH, "grid_search_y_pred.npy"), gs_preds)
    np.save(os.path.join(OUT_PATH, "y_test.npy"), y_test)

    gs_metrics = get_metrics(gs_preds, y_test, task_type=argsdict["task_type"])

    np.save(os.path.join(OUT_PATH, "grid_search_metrics.npy"), gs_metrics)

    # Default parameters
    print("Training with default parameters...")
    preds, estimator = train_model(
        model,
        x_train,
        y_train,
        x_test,
        parameters=None,
        model_type=argsdict["rf_or_svm"],
    )
    print("Finished training with default parameters...")
    save_joblib(estimator, "default_estimator", OUT_PATH)

    if argsdict["task_type"] == "regression":
        preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

    np.save(os.path.join(OUT_PATH, "default_y_pred.npy"), preds)

    default_metrics = get_metrics(preds, y_test, task_type=argsdict["task_type"])

    np.save(os.path.join(OUT_PATH, "default_metrics.npy"), default_metrics)


if __name__ == "__main__":
    main()
