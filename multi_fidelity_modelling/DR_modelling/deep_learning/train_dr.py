import argparse
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pathlib import Path

from multi_fidelity_modelling.src.graph_models import Estimator
from multi_fidelity_modelling.src.data_loading import GeometricDataModule


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--target-label', required=True)
    parser.add_argument('--node-latent-dim', type=int, required=True)
    parser.add_argument('--graph-latent-dim', type=int, required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--smiles-column', required=True)
    parser.add_argument('--max-atomic-number', type=int)
    parser.add_argument('--readout', choices=['linear', 'global_mean_pool', 'global_add_pool', 'global_max_pool', 'set_transformer'])
    parser.add_argument('--id-column', required=True)
    parser.add_argument('--use-vgae', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--conv', choices=['GCN', 'GIN', 'PNA'])
    parser.add_argument('--use-batch-norm', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--gnn-intermediate-dim', type=int, required=True)
    parser.add_argument('--edge-dim', type=int, required=False, default=None)
    parser.add_argument('--name', type=str)
    parser.add_argument('--lbl-or-emb', type=str)
    parser.add_argument('--auxiliary-dim', type=int)
    parser.add_argument('--task-type', type=str)
    parser.add_argument('--auxiliary-data-column-name', type=str)
    parser.add_argument('--use-cuda', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--ckpt-path', type=str, required=False, default=None)

    args = parser.parse_args()
    argsdict = vars(args)

    assert(argsdict['lbl_or_emb'] in ['lbl', 'emb', None])


    LR = 0.0001
    BATCH_SIZE = 32
    SEED = 0
    MAX_NUM_ATOMS_IN_MOL = 125

    max_atom_num = argsdict['max_atomic_number']
    num_atom_features = max_atom_num + 27
    task_type = argsdict['task_type']
    gnn_type = 'VGAE' if argsdict['use_vgae'] else 'GNN'
    edges = 'edges=True' if argsdict['edge_dim'] else 'edges=False'

    pl.seed_everything(SEED)

    ############## Model set-up ##############
    use_standard_scaler = True
    data_module = GeometricDataModule(batch_size=BATCH_SIZE, seed=SEED, max_atom_num=max_atom_num,
                                      train_path=os.path.join(argsdict['data_path'], 'train.csv'), num_cores=(0, 0, 0),
                                      smiles_column_name=argsdict['smiles_column'],
                                      use_standard_scaler=use_standard_scaler,
                                      id_column=argsdict['id_column'], split_train=False,
                                      label_column_name=argsdict['target_label'],
                                      lbl_or_emb=argsdict['lbl_or_emb'],
                                      separate_valid_path=os.path.join(argsdict['data_path'], 'validate.csv'),
                                      separate_test_path=os.path.join(argsdict['data_path'], 'test.csv'),
                                      auxiliary_data_column_name=argsdict['auxiliary_data_column_name'])

    data_module.prepare_data()
    data_module.setup()

    Path(argsdict['out_dir']).mkdir(exist_ok=True, parents=True)
    MONITOR_LOSS = 'val_total_loss'

    # Logging
    csv_logger = CSVLogger(argsdict['out_dir'], name='log')

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=MONITOR_LOSS,
        dirpath=argsdict['out_dir'],
        filename='%s-%s-%s-%s-%s-%s-num_layers=%s-batch_norm=%s-{epoch:03d}-{val_total_loss:.5f}' % (task_type, gnn_type, argsdict['conv'], edges, argsdict['readout'], argsdict['target_label'], argsdict['num_layers'], argsdict['use_batch_norm']),
        mode='min',
        save_top_k=1,
    )

    scaler = None
    if use_standard_scaler:
        scaler = data_module.get_scaler()

    gnn_args = dict(task_type=task_type, conv_type=argsdict['conv'], readout=argsdict['readout'], num_features=num_atom_features, node_latent_dim=argsdict['node_latent_dim'], batch_size=BATCH_SIZE, lr=LR, linear_output_size=1,  max_num_atoms_in_mol=MAX_NUM_ATOMS_IN_MOL, scaler=scaler, graph_latent_dim=argsdict['graph_latent_dim'], monitor_loss=MONITOR_LOSS, gnn_intermediate_dim=argsdict['gnn_intermediate_dim'], num_layers=argsdict['num_layers'], use_batch_norm=argsdict['use_batch_norm'], use_vgae=argsdict['use_vgae'], auxiliary_dim=argsdict['auxiliary_dim'])

    if argsdict['edge_dim']:
        gnn_args = gnn_args | dict(edge_dim=argsdict['edge_dim'])

    if argsdict['conv'] == 'PNA':
        pna_args = dict(train_dataset=data_module.dataset, name=argsdict['name'])
        gnn_args = gnn_args | pna_args


    model = Estimator(**gnn_args)

    trainer_common_args = dict(devices=1, callbacks=[EarlyStopping(monitor=MONITOR_LOSS, mode='min', patience=30), checkpoint_callback], logger=csv_logger)

    if argsdict['use_cuda']:
        model = model.cuda()
        trainer_args = trainer_common_args | dict(accelerator='gpu',  strategy=pl.plugins.training_type.SingleDevicePlugin(torch.device(f'cuda:{torch.cuda.current_device()}')))
    else:
        trainer_args = trainer_common_args | dict(accelerator='cpu',  strategy=pl.plugins.training_type.SingleDevicePlugin(torch.device('cpu')))

    # Train
    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    
    ### Save outputs
    np.save(f'{argsdict["out_dir"]}/test_metrics.npy', model.test_metrics)
    np.save(f'{argsdict["out_dir"]}/test_output.npy', model.test_output)
    np.save(f'{argsdict["out_dir"]}/test_true.npy', model.test_true)


if __name__ == "__main__":
    main()