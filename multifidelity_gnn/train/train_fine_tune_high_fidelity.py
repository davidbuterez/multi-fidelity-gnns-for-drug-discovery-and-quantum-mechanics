import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Imports from this project
from ..src.graph_models import Estimator
from ..src.data_loading import GeometricDataModule


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description="Fine-tuning on high-fidelity script.")
    parser.add_argument("--high-fidelity-data-path", required=True)
    parser.add_argument("--high-fidelity-label", required=True)
    parser.add_argument("--node-latent-dim", type=int, required=True)
    parser.add_argument("--graph-latent-dim", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smiles-column", required=True)
    parser.add_argument("--max-atomic-number", type=int, required=True)
    parser.add_argument("--max-num-atoms-in-mol", type=int, required=True)
    parser.add_argument(
        "--readout",
        choices=[
            "linear",
            "global_mean_pool",
            "global_add_pool",
            "global_max_pool",
            "set_transformer",
        ],
        required=True,
    )
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--use-vgae", action=argparse.BooleanOptionalAction, required=True, default=True)
    parser.add_argument("--num-layers", type=int, required=True, default=3)
    parser.add_argument("--conv", choices=["GCN", "GIN", "PNA"], required=True)
    parser.add_argument("--use-batch-norm", action=argparse.BooleanOptionalAction, required=True, default=True)
    parser.add_argument("--linear-interim-dim", type=int, required=False)
    parser.add_argument("--linear-dropout-p", type=float, required=False)
    parser.add_argument("--set-transformer-hidden-dim", type=int, required=False)
    parser.add_argument("--set-transformer-num-heads", type=int, required=False)
    parser.add_argument("--set-transformer-num-sabs", type=int, required=False)
    parser.add_argument("--set-transformer-dropout", type=float, required=False)
    parser.add_argument("--dataloader-num-workers", type=int, required=False, default=12)
    parser.add_argument("--gnn-intermediate-dim", type=int, required=True)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, required=False, default=False)
    parser.add_argument("--edge-dim", type=int, required=False, default=None)
    parser.add_argument("--logging-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--low-fidelity-ckpt-path", type=str, required=False, default=None)
    parser.add_argument("--freeze-vgae", action=argparse.BooleanOptionalAction, required=False, default=False)

    args = parser.parse_args()
    argsdict = vars(args)

    SEED = 0
    LR = 0.00005

    batch_size = argsdict["batch_size"]
    max_num_atoms_in_mol = argsdict["max_num_atoms_in_mol"]
    max_atom_num = argsdict["max_atomic_number"]
    num_atom_features = max_atom_num + 27
    task_type = "regression"
    gnn_type = "VGAE" if argsdict["use_vgae"] else "GNN"

    set_transformer_hidden_dim = argsdict["set_transformer_hidden_dim"]
    set_transformer_num_heads = argsdict["set_transformer_num_heads"]
    set_transformer_num_sabs = argsdict["set_transformer_num_sabs"]
    set_transformer_dropout = argsdict["set_transformer_dropout"]

    freeze_vgae = argsdict['freeze_vgae']

    pl.seed_everything(SEED)

    ############## Model set-up ##############
    use_standard_scaler = True
    data_module = GeometricDataModule(
        seed=SEED,
        batch_size=batch_size,
        max_atom_num=max_atom_num,
        train_path=os.path.join(argsdict["high_fidelity_data_path"], "train.csv"),
        num_cores=tuple([argsdict["dataloader_num_workers"]] * 3),
        smiles_column_name=argsdict["smiles_column"],
        use_standard_scaler=use_standard_scaler,
        id_column=argsdict["id_column"],
        label_column_name=argsdict["high_fidelity_label"],
        separate_valid_path=os.path.join(argsdict["high_fidelity_data_path"], "validate.csv"),
        separate_test_path=os.path.join(argsdict["high_fidelity_data_path"], "test.csv"),
    )

    data_module.prepare_data()
    data_module.setup()

    MONITOR_LOSS = "val_total_loss"

    NAME = f"{argsdict['logging_name']}+{argsdict['conv']}+{argsdict['readout']}+{argsdict['num_layers']}"
    NAME += f"+bn={argsdict['use_batch_norm']}+gnn_type={gnn_type}+LR={LR}+graph_dim={argsdict['graph_latent_dim']}"

    if argsdict["readout"] == "linear":
        NAME += f"+linear_interim_dim={argsdict['linear_interim_dim']}+linear_dropout_p={argsdict['linear_dropout_p']}"

    if argsdict["readout"] == "set_transformer":
        NAME += f"st_hidden_dim={set_transformer_hidden_dim}+st_num_heads={set_transformer_num_heads}"
        NAME += f"+st_num_sabs={set_transformer_num_sabs}+st_dropout={set_transformer_dropout}+st_bn=True"

    scaler = None
    if use_standard_scaler:
        scaler = data_module.get_scaler()

    gnn_args = dict(
        task_type=task_type,
        conv_type=argsdict["conv"],
        readout=argsdict["readout"],
        num_features=num_atom_features,
        node_latent_dim=argsdict["node_latent_dim"],
        batch_size=batch_size,
        lr=LR,
        linear_output_size=1,
        max_num_atoms_in_mol=max_num_atoms_in_mol,
        scaler=scaler,
        graph_latent_dim=argsdict["graph_latent_dim"],
        monitor_loss=MONITOR_LOSS,
        gnn_intermediate_dim=argsdict["gnn_intermediate_dim"],
        num_layers=argsdict["num_layers"],
        use_batch_norm=argsdict["use_batch_norm"],
        use_vgae=argsdict["use_vgae"],
        name=argsdict["logging_name"],
        linear_interim_dim=argsdict["linear_interim_dim"],
        linear_dropout_p=argsdict["linear_dropout_p"],
        set_transformer_hidden_dim=set_transformer_hidden_dim,
        set_transformer_num_heads=set_transformer_num_heads,
        set_transformer_num_sabs=set_transformer_num_sabs,
        set_transformer_dropout=set_transformer_dropout,
        only_train=True,
    )

    if argsdict["edge_dim"]:
        gnn_args = gnn_args | dict(edge_dim=argsdict["edge_dim"])

    if argsdict["conv"] == "PNA":
        pna_args = dict(train_dataset=data_module.dataset)
        gnn_args = gnn_args | pna_args

    model = Estimator.load_from_checkpoint(argsdict["low_fidelity_ckpt_path"], **gnn_args)

    if freeze_vgae:
        print("Freezing VGAE layers!")
        for param in model.gnn_model.parameters():
            param.requires_grad = False
    else:
        print("Not freezing any part of the network!")

    if argsdict["use_cuda"]:
        accelerator = dict(accelerator="gpu")
    else:
        accelerator = dict(accelerator="cpu")

    # Logging
    logger = WandbLogger(
        name=NAME,
        project="Multi-fidelity modelling | Fine-tune on high-fidelity",
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR_LOSS,
        dirpath=argsdict["out_dir"],
        filename="%s-%s-%s-%s-num_layers=%s-batch_norm=%s-{epoch:03d}-{val_total_loss:.5f}"
        % (
            task_type,
            gnn_type,
            argsdict["conv"],
            argsdict["high_fidelity_label"],
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
        max_epochs=-1,
        min_epochs=1
    )

    trainer_args = common_trainer_args | accelerator

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    np.save(os.path.join(argsdict["out_dir"], "test_metrics.npy"), model.test_metrics)
    np.save(os.path.join(argsdict["out_dir"], "test_preds.npy"), model.test_output)
    np.save(os.path.join(argsdict["out_dir"], "test_true.npy"), model.test_true)


if __name__ == "__main__":
    main()
