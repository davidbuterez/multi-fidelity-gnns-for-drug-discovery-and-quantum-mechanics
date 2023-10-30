import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Imports from this project
from ..src.graph_models import Estimator
from ..src.data_loading import GeometricDataModule


def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description="Low-fidelity training script.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--low-fidelity-label", required=True)
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
    parser.add_argument("--monitor-loss-name", type=str, required=True)
    parser.add_argument(
        "--use-vgae", action=argparse.BooleanOptionalAction, required=True, default=True
    )
    parser.add_argument("--num-layers", type=int, required=True, default=3)
    parser.add_argument("--conv", choices=["GCN", "GIN", "PNA"], required=True)
    parser.add_argument(
        "--use-batch-norm",
        action=argparse.BooleanOptionalAction,
        required=True,
        default=True,
    )
    parser.add_argument("--linear-interim-dim", type=int, required=False)
    parser.add_argument("--linear-dropout-p", type=float, required=False)
    parser.add_argument("--set-transformer-hidden-dim", type=int, required=False)
    parser.add_argument("--set-transformer-num-heads", type=int, required=False)
    parser.add_argument("--set-transformer-num-sabs", type=int, required=False)
    parser.add_argument("--set-transformer-dropout", type=float, required=False)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument(
        "--dataloader-num-workers", type=int, required=False, default=12
    )
    parser.add_argument("--gnn-intermediate-dim", type=int, required=True)
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )
    parser.add_argument("--edge-dim", type=int, required=False, default=None)
    parser.add_argument("--logging-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ckpt-path", type=str, required=False, default=None)
    parser.add_argument(
        "--load-ckpt-and-generate-embs",
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False,
    )

    args = parser.parse_args()
    argsdict = vars(args)

    should_load_and_generate_embs = False
    if (
        argsdict["load_ckpt_and_generate_embs"] is not None
        and argsdict["load_ckpt_and_generate_embs"]
    ):
        assert argsdict["ckpt_path"] is not None
        should_load_and_generate_embs = True

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

    pl.seed_everything(SEED)

    ############## Model set-up ##############
    use_standard_scaler = True
    data_module = GeometricDataModule(
        seed=SEED,
        batch_size=batch_size,
        max_atom_num=max_atom_num,
        train_path=argsdict["data_path"],
        num_cores=tuple([argsdict["dataloader_num_workers"]] * 3),
        smiles_column_name=argsdict["smiles_column"],
        use_standard_scaler=use_standard_scaler,
        id_column=argsdict["id_column"],
        label_column_name=argsdict["low_fidelity_label"],
        separate_valid_path=None,
        separate_test_path=argsdict["data_path"]
        if should_load_and_generate_embs
        else None,
    )

    data_module.prepare_data()
    data_module.setup()

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
        monitor_loss=argsdict["monitor_loss_name"],
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

    if should_load_and_generate_embs:
        model = Estimator.load_from_checkpoint(argsdict["ckpt_path"], **gnn_args)
    else:
        model = Estimator(**gnn_args)

    if argsdict["use_cuda"]:
        model = model.cuda()
        accelerator = dict(accelerator="gpu")
    else:
        accelerator = dict(accelerator="cpu")

    if not should_load_and_generate_embs:
        # Logging
        logger = WandbLogger(
            name=NAME,
            project="Multi-fidelity modelling | Low-fidelity training",
        )

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor=argsdict["monitor_loss_name"],
            dirpath=argsdict["out_dir"],
            filename="{epoch:03d}-{train_total_loss:.5f}",
            mode="min",
            save_top_k=-1,
            every_n_train_steps=0,
            every_n_epochs=1,
            train_time_interval=None,
            save_on_train_epoch_end=True,
        )

        common_trainer_args = dict(
            callbacks=[checkpoint_callback],
            logger=logger,
            min_epochs=argsdict["num_epochs"],
            max_epochs=argsdict["num_epochs"] + 1,
            devices=1,
            num_sanity_val_steps=0,
        )

        trainer_args = common_trainer_args | accelerator

    if should_load_and_generate_embs:
        trainer = pl.Trainer(devices=1, **accelerator)
        trainer.test(model=model, datamodule=data_module)

        np.save(
            os.path.join(argsdict["out_dir"], "low_fidelity_predictions.npy"),
            model.test_output,
        )
        np.save(
            os.path.join(argsdict["out_dir"], "low_fidelity_true.npy"), model.test_true
        )
        np.save(
            os.path.join(argsdict["out_dir"], "low_fidelity_graph_embeddings.npy"),
            model.test_graph_embeddings,
        )

    else:
        if argsdict["ckpt_path"]:
            trainer_args = (
                dict(resume_from_checkpoint=argsdict["ckpt_path"]) | trainer_args
            )

        trainer = pl.Trainer(**trainer_args)
        trainer.fit(model=model, train_dataloaders=data_module.train_dataloader())


if __name__ == "__main__":
    main()
