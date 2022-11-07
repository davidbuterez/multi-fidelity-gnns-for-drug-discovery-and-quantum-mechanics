import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Imports from this project
from multi_fidelity_modelling.src.graph_models import Estimator
from multi_fidelity_modelling.src.data_loading import GeometricDataModule


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description='SD training script.')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--sd-label', required=True)
    parser.add_argument('--node-latent-dim', type=int, required=True)
    parser.add_argument('--graph-latent-dim', type=int, required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--smiles-column', required=True)
    parser.add_argument('--max-atomic-number', type=int)
    parser.add_argument('--readout', choices=['linear', 'global_mean_pool', 'global_add_pool', 'global_max_pool', 'set_transformer'])
    parser.add_argument('--id-column', required=True)

    parser.add_argument('--monitor-loss-name', type=str, required=True)
    parser.add_argument('--use-vgae', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--conv', choices=['GCN', 'GIN', 'PNA'])
    parser.add_argument('--use-batch-norm', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-epochs', type=int, required=True)
    parser.add_argument('--gnn-intermediate-dim', type=int, required=True)
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--edge-dim', type=int, required=False, default=None)
    parser.add_argument('--name', type=str)

    parser.add_argument('--ckpt-path', type=str, required=False, default=None)

    args = parser.parse_args()
    argsdict = vars(args)

    LR = 0.0001
    BATCH_SIZE = 32
    SEED = 0
    MAX_NUM_ATOMS_IN_MOL = 125

    max_atom_num = argsdict['max_atomic_number']
    num_atom_features = max_atom_num + 27
    task_type = 'regression'
    gnn_type = 'VGAE' if argsdict['use_vgae'] else 'GNN'
    edges = 'edges=True' if argsdict['edge_dim'] else 'edges=False'

    pl.seed_everything(SEED)


    ############## Model set-up ##############
    use_standard_scaler = True
    data_module = GeometricDataModule(batch_size=BATCH_SIZE, seed=SEED, max_atom_num=max_atom_num,\
                                      train_path=argsdict['data_path'], num_cores=(8, 0, 0),
                                      smiles_column_name=argsdict['smiles_column'], \
                                      use_standard_scaler=use_standard_scaler, \
                                      id_column=argsdict['id_column'], split_train=False,
                                      label_column_name=argsdict['sd_label'])

    data_module.prepare_data()
    data_module.setup()

    # Logging
    csv_logger = CSVLogger(argsdict['out_dir'], name='{}-{}-{}-{}-{}-{}-num_layers={}-batch_norm={}-logs'.format(task_type, gnn_type, argsdict['conv'], edges, argsdict['readout'], argsdict['sd_label'], argsdict['num_layers'], argsdict['use_batch_norm']))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=argsdict['monitor_loss_name'],
        dirpath=argsdict['out_dir'],
        filename='%s-%s-%s-%s-%s-%s-num_layers=%s-batch_norm=%s-{epoch:03d}-{train_total_loss:.5f}' % (task_type, gnn_type, argsdict['conv'], edges, argsdict['readout'], argsdict['sd_label'], argsdict['num_layers'], argsdict['use_batch_norm']),
        mode='min',
        save_top_k=-1,
        every_n_train_steps=0,
        every_n_epochs=1, 
        train_time_interval=None,
        save_on_train_epoch_end=True
    )

    scaler = None
    if use_standard_scaler:
        scaler = data_module.get_scaler()

    gnn_args = dict(task_type=task_type, conv_type=argsdict['conv'], readout=argsdict['readout'], num_features=num_atom_features, \
                        node_latent_dim=argsdict['node_latent_dim'], batch_size=BATCH_SIZE, lr=LR, linear_output_size=1, \
                        max_num_atoms_in_mol=MAX_NUM_ATOMS_IN_MOL, scaler=scaler, graph_latent_dim=argsdict['graph_latent_dim'], \
                        monitor_loss=argsdict['monitor_loss_name'], gnn_intermediate_dim=argsdict['gnn_intermediate_dim'], \
                        num_layers=argsdict['num_layers'], use_batch_norm=argsdict['use_batch_norm'], use_vgae=argsdict['use_vgae'])

    if argsdict['edge_dim']:
        gnn_args = gnn_args | dict(edge_dim=argsdict['edge_dim'])

    if argsdict['conv'] == 'PNA':
        pna_args = dict(train_dataset=data_module.dataset, name=argsdict['name'])
        gnn_args = gnn_args | pna_args


    model = Estimator(**gnn_args)

    common_trainer_args = dict(callbacks=[checkpoint_callback], logger=csv_logger, min_epochs=argsdict['num_epochs'], 
                                max_epochs=argsdict['num_epochs'] + 1, num_sanity_val_steps=0, devices=1, check_val_every_n_epoch=1000)

    if argsdict['use_cuda']:
        model = model.cuda()
        trainer_args = common_trainer_args | dict(accelerator='gpu', strategy=pl.plugins.training_type.SingleDevicePlugin(torch.device(f'cuda:{torch.cuda.current_device()}')))
    else:
        trainer_args = common_trainer_args | dict(accelerator='cpu', strategy=pl.plugins.training_type.SingleDevicePlugin('cpu'))

    if argsdict['ckpt_path']:
        trainer_args = dict(resume_from_checkpoint=argsdict['ckpt_path']) | trainer_args

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()