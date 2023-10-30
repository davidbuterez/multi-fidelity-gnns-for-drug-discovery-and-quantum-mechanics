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
from .graph_models_fidelity_emb import Estimator
from .data_loading import MixedFidelityGeometricDataModule


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--low-fi-data-path", required=False)
    parser.add_argument("--high-fi-train-data-path", required=True)
    parser.add_argument("--high-fi-val-data-path", required=True)
    parser.add_argument("--high-fi-test-data-path", required=True)
    parser.add_argument("--low-fi-batch-size", type=int, default=512, required=False)
    parser.add_argument("--high-fi-batch-size", type=int, default=32, required=True)
    parser.add_argument("--low-fi-target-label", required=False)
    parser.add_argument("--high-fi-target-label", required=True)
    parser.add_argument("--just-high-fi", required=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--max-atomic-number", type=int, required=True)
    parser.add_argument("--edge-dim", type=int, required=False)
    parser.add_argument("--node-latent-dim", type=int, required=True)
    parser.add_argument("--edge-latent-dim", type=int, required=True)
    parser.add_argument("--fidelity-embedding-dim", type=int, required=True)
    parser.add_argument("--use-set2set", action=argparse.BooleanOptionalAction, required=True, default=True)

    parser.add_argument("--smiles-column", required=True)
    parser.add_argument("--id-column", required=True)

    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--logging-name", type=str, required=True)
    parser.add_argument("--task-type", choices=["regression", "classification"], type=str, required=True)

    parser.add_argument("--dataloader-num-workers", type=int, required=False, default=0)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, required=False, default=False)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--num-layers", type=int, required=False)
    parser.add_argument("--use-batch-norm", action=argparse.BooleanOptionalAction, required=False, default=False)

    parser.add_argument("--ckpt-path", type=str)

    args = parser.parse_args()
    argsdict = vars(args)

    SEED = 0
    LR = argsdict["lr"]

    lf_batch_size = argsdict["low_fi_batch_size"]
    hf_batch_size = argsdict["high_fi_batch_size"]

    lf_target = argsdict["low_fi_target_label"]
    hf_target = argsdict["high_fi_target_label"]

    max_atom_num = argsdict["max_atomic_number"]
    num_atom_features = max_atom_num + 27
    task_type = argsdict["task_type"]
    edge_dim = argsdict["edge_dim"]
    node_latent_dim = argsdict["node_latent_dim"]
    edge_latent_dim = argsdict["edge_latent_dim"]
    fidelity_embedding_dim = argsdict["fidelity_embedding_dim"]
    use_set2set = argsdict["use_set2set"]
    just_hf = argsdict["just_high_fi"]

    pl.seed_everything(SEED)

    ############## Model set-up ##############
    use_standard_scaler = True
    data_module = MixedFidelityGeometricDataModule(
        low_fidelity_path=argsdict["low_fi_data_path"],
        high_fidelity_train_path=argsdict["high_fi_train_data_path"],
        high_fidelity_valid_path=argsdict["high_fi_val_data_path"],
        high_fidelity_test_path=argsdict["high_fi_test_data_path"],
        lf_batch_size=lf_batch_size,
        hf_batch_size=hf_batch_size,
        low_fidelity_target_label=lf_target,
        high_fidelity_target_label=hf_target,
        max_atom_num=max_atom_num,
        id_column=argsdict["id_column"],
        smiles_column=argsdict["smiles_column"],
        use_standard_scaler=use_standard_scaler,
        num_cores=tuple([argsdict["dataloader_num_workers"]] * 3),
        just_hf=just_hf
    )

    data_module.prepare_data()
    data_module.setup()

    Path(argsdict["out_dir"]).mkdir(exist_ok=True, parents=True)

    if not just_hf:
        MONITOR_LOSS = "sd_train_loss"
    else:
        MONITOR_LOSS = "val_loss"

    # Logging
    logger = WandbLogger(
        project="MFSE",
        name=argsdict["logging_name"],
    )

    lf_scaler, hf_scaler = None, None
    if use_standard_scaler:
        lf_scaler = data_module.get_lf_scaler()
        hf_scaler = data_module.get_hf_scaler()

    gnn_args = dict(
        task_type=task_type,
        num_features=num_atom_features,
        edge_dim=edge_dim,
        node_latent_dim=node_latent_dim,
        edge_latent_dim=edge_latent_dim,
        fidelity_embedding_dim=fidelity_embedding_dim,
        use_set2set=use_set2set,
        lf_batch_size=lf_batch_size,
        hf_batch_size=hf_batch_size,
        lr=LR,
        linear_output_size=1,
        lf_scaler=lf_scaler,
        hf_scaler=hf_scaler,
        monitor_loss=MONITOR_LOSS,
        num_layers=argsdict["num_layers"],
        use_batch_norm=argsdict["use_batch_norm"],
        just_hf=just_hf,
    )

    model = Estimator(**gnn_args)

    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR_LOSS,
        dirpath=argsdict["out_dir"],
        filename="{epoch:03d}-{sd_train_loss:.5f}",
        mode="min",
        save_top_k=1,
    )

    if just_hf:
        early_stopping_callback = EarlyStopping(
            monitor=MONITOR_LOSS, mode="min", patience=100
        )

    callbacks = (
        [checkpoint_callback]
        if not just_hf
        else [checkpoint_callback, early_stopping_callback]
    )

    common_trainer_args = dict(
        devices=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=201 if not just_hf else -1,
        min_epochs=200 if not just_hf else 1,
        log_every_n_steps=10,
    )

    # Train
    if argsdict["use_cuda"]:
        model = model.cuda()
        accelerator = dict(accelerator="gpu")
    else:
        accelerator = dict(accelerator="cpu")

    trainer_args = common_trainer_args | accelerator

    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, data_module, ckpt_path=argsdict["ckpt_path"])
    trainer.test(model, data_module)

    np.save(os.path.join(argsdict["out_dir"], "test_metrics.npy"), model.test_metrics)
    np.save(os.path.join(argsdict["out_dir"], "test_preds.npy"), model.test_output)
    np.save(os.path.join(argsdict["out_dir"], "test_true.npy"), model.test_true)


if __name__ == "__main__":
    main()
