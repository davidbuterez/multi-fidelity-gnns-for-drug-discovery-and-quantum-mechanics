import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path

# Imports from this project
from ..src.graph_models import Estimator
from ..src.data_loading import GeometricDataModule


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--node-latent-dim", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smiles-column", required=True)
    parser.add_argument("--max-atomic-number", type=int, required=True)
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--use-vgae", action=argparse.BooleanOptionalAction, required=True, default=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--conv", choices=["GCN", "GIN", "PNA"], type=str, required=True)
    parser.add_argument("--use-batch-norm", action=argparse.BooleanOptionalAction, required=True, default=True)
    parser.add_argument("--dataloader-num-workers", type=int, required=False, default=0)
    parser.add_argument("--gnn-intermediate-dim", type=int, required=True)
    parser.add_argument("--edge-dim", type=int, required=False)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, required=False, default=False)
    parser.add_argument("--logging-name", type=str, required=True)
    parser.add_argument("--lbl-or-emb", type=str, required=False)
    parser.add_argument("--auxiliary-dim", type=int, required=False)
    parser.add_argument("--task-type", choices=["regression", "classification"], type=str, required=True)
    parser.add_argument("--train-auxiliary-data-column-name", type=str, required=False)
    parser.add_argument("--eval-auxiliary-data-column-name", type=str, required=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-num-atoms-in-mol", type=int, required=True)

    args = parser.parse_args()
    argsdict = vars(args)

    assert argsdict["lbl_or_emb"] in ["lbl", "emb", None]

    SEED = 0
    LR = 0.0001

    batch_size = argsdict["batch_size"]
    max_num_atoms_in_mol = argsdict["max_num_atoms_in_mol"]
    max_atom_num = argsdict["max_atomic_number"]
    num_atom_features = max_atom_num + 27
    task_type = argsdict["task_type"]
    gnn_type = "VGAE" if argsdict["use_vgae"] else "GNN"
    edges = "edges=True" if argsdict["edge_dim"] else "edges=False"

    pl.seed_everything(SEED)

    ############## Model set-up ##############
    use_standard_scaler = True
    data_module = GeometricDataModule(
        batch_size=batch_size,
        seed=SEED,
        max_atom_num=max_atom_num,
        train_path=os.path.join(argsdict["data_path"], "train.csv"),
        separate_valid_path=os.path.join(argsdict["data_path"], "validate.csv"),
        separate_test_path=os.path.join(argsdict["data_path"], "test.csv"),
        num_cores=tuple([argsdict["dataloader_num_workers"]] * 3),
        smiles_column_name=argsdict["smiles_column"],
        use_standard_scaler=use_standard_scaler,
        id_column=argsdict["id_column"],
        label_column_name=argsdict["target_label"],
        lbl_or_emb=argsdict["lbl_or_emb"],
        train_auxiliary_data_column_name=argsdict["train_auxiliary_data_column_name"],
        eval_auxiliary_data_column_name=argsdict["eval_auxiliary_data_column_name"],
    )

    data_module.prepare_data()
    data_module.setup()

    Path(argsdict["out_dir"]).mkdir(exist_ok=True, parents=True)
    MONITOR_LOSS = "val_total_loss"

    # Logging
    logger = WandbLogger(
        project="Multi-fidelity modelling | High-fidelity training",
        name=argsdict["logging_name"],
    )

    scaler = None
    if use_standard_scaler:
        scaler = data_module.get_scaler()

    gnn_args = dict(
        task_type=task_type,
        conv_type=argsdict["conv"],
        num_features=num_atom_features,
        node_latent_dim=argsdict["node_latent_dim"],
        batch_size=batch_size,
        lr=LR,
        linear_output_size=1,
        max_num_atoms_in_mol=max_num_atoms_in_mol,
        scaler=scaler,
        monitor_loss=MONITOR_LOSS,
        gnn_intermediate_dim=argsdict["gnn_intermediate_dim"],
        num_layers=argsdict["num_layers"],
        use_batch_norm=argsdict["use_batch_norm"],
        use_vgae=argsdict["use_vgae"],
        auxiliary_dim=argsdict["auxiliary_dim"],
        only_train=False,
    )

    if argsdict["edge_dim"]:
        gnn_args = gnn_args | dict(edge_dim=argsdict["edge_dim"])

    if argsdict["conv"] == "PNA":
        pna_args = dict(
            train_dataset=data_module.dataset, name=argsdict["logging_name"]
        )
        gnn_args = gnn_args | pna_args

    model = Estimator(**gnn_args)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR_LOSS,
        dirpath=argsdict["out_dir"],
        filename="%s-%s-%s-%s-%s-num_layers=%s-batch_norm=%s-{epoch:03d}-{val_total_loss:.5f}"
        % (
            task_type,
            gnn_type,
            argsdict["conv"],
            edges,
            argsdict["target_label"],
            argsdict["num_layers"],
            argsdict["use_batch_norm"],
        ),
        mode="min",
        save_top_k=1,
    )

    early_stopping_callback = EarlyStopping(
        monitor=MONITOR_LOSS, mode="min", patience=100
    )

    common_trainer_args = dict(
        devices=1,
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )

    # Train
    if argsdict["use_cuda"]:
        accelerator = dict(accelerator="gpu")
    else:
        accelerator = dict(accelerator="cpu")

    trainer_args = common_trainer_args | accelerator

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    np.save(os.path.join(argsdict["out_dir"], "test_metrics.npy"), model.test_metrics)
    np.save(os.path.join(argsdict["out_dir"], "test_preds.npy"), model.test_output)
    np.save(os.path.join(argsdict["out_dir"], "test_true.npy"), model.test_true)


if __name__ == "__main__":
    main()
