import argparse
import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader as GeometricDataLoader
from pathlib import Path

# Imports from this project
from pathlib import Path
# import sys
# path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
# sys.path.insert(0, path)

from .load_QM7 import load_QM7_aux
from .lightning_wrap import Estimator


MAX_NUM_ATOMS_IN_MOL = {
    'QM7': 23,
    'QM8': 26,
    'QM9': 29,
    'QMugs': 228,
    'benzene': 12,
    'aspirin': 21,
    'malonaldehyde': 9,
    'ethanol': 9,
    'toluene': 15
}

TARGET_ID_TO_PROPERTY = {
    0: 'First excitation energy (ZINDO)',
    1: 'Electron affinity (ZINDO/s)',
    2: 'Excitation energy at maximal absorption (ZINDO)',
    3: 'Atomization energy (DFT/PBE0)',
    4: 'Highest occupied molecular orbital (GW)',
    5: 'Highest occupied molecular orbital (PBE0)',
    6: 'Highest occupied molecular orbital (ZINDO/s)',
    7: 'Maximal absorption intensity (ZINDO)',
    8: 'Ionization potential (ZINDO/s)',
    9: 'Lowest unoccupied molecular orbital (GW)',
    10: 'Lowest unoccupied molecular orbital (PBE0)',
    11: 'Lowest unoccupied molecular orbital (ZINDO/s)',
    12: 'Polarizability (self-consistent screening)',
    13: 'Polarizability (DFT/PBE0)'
}

def main():
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dataset-download-dir', type=str, required=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--readout', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr', type=float, default=0.0001, required=False)
    parser.add_argument('--random-seed', type=int, required=True)
    parser.add_argument('--target-id', type=int, required=True)

    parser.add_argument('--set-transformer-hidden-dim', type=int, default=512, required=False)
    parser.add_argument('--set-transformer-num-heads', type=int, default=16, required=False)
    parser.add_argument('--set-transformer-num-sabs', type=int, default=2, required=False)

    parser.add_argument('--schnet-hidden-channels', type=int, default=128)
    parser.add_argument('--schnet-num-filters', type=int, default=128)
    parser.add_argument('--schnet-num-interactions', type=int, default=6)

    parser.add_argument('--ckpt-path', type=str, required=False, default=None)

    parser.add_argument('--lbl-or-emb', type=str)
    parser.add_argument('--include', type=str)
    parser.add_argument('--emb-type', type=str)
    parser.add_argument('--aux-dim', type=int)

    args = parser.parse_args()
    argsdict = vars(args)

    SEED = 0
    learning_rate = argsdict['lr']
    batch_size = argsdict['batch_size']

    schnet_hidden_channels = argsdict['schnet_hidden_channels']
    schnet_num_filters = argsdict['schnet_num_filters']
    schnet_num_interactions = argsdict['schnet_num_interactions']

    set_transformer_hidden_dim = argsdict['set_transformer_hidden_dim']
    set_transformer_num_heads = argsdict['set_transformer_num_heads']
    set_transformer_num_sabs = argsdict['set_transformer_num_sabs']

    random_seed = argsdict['random_seed']
    target_id = argsdict['target_id']
    target_name = TARGET_ID_TO_PROPERTY[target_id]
    readout = argsdict['readout']

    lbl_or_emb = argsdict['lbl_or_emb']
    include = argsdict['include']
    emb_type = argsdict['emb_type']
    aux_dim = argsdict['aux_dim']

    pl.seed_everything(SEED)

    dataset = argsdict['dataset']
    assert dataset  == 'QM7'
    assert target_id in [4, 9]
    print('Loading dataset...')

    NUM_WORKERS = 6

    train, val, test, scaler = load_QM7_aux(random_seed, target_id)

    train_dataloader = GeometricDataLoader(train, batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = GeometricDataLoader(val, batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader = GeometricDataLoader(test, batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print('Loaded dataset!')

    NAME = f'DATASET={dataset}+trg_name={target_name}+rs={random_seed}+rd={readout}+lr={learning_rate}'
    NAME += f'+sch_hd_ch={schnet_hidden_channels}+sch_num_intr={schnet_num_interactions}'
    NAME += f'+sch_num_filt={schnet_num_filters}'

    NAME += f'+lbl_or_emb={lbl_or_emb}+incl={include}+emb_type={emb_type}'

    if readout == 'set_transformer':
        NAME += f'+st_num_SABs={set_transformer_num_sabs}+st_hidden_dim={set_transformer_hidden_dim}'
        NAME += f'+st_num_heads={set_transformer_num_heads}'


    OUT_DIR = os.path.join(argsdict['out_dir'], f'SchNet/{dataset}/{target_id}/{readout}', NAME)
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    gnn_args = dict(
        readout=readout, batch_size=batch_size, lr=learning_rate,
        max_num_atoms_in_mol=MAX_NUM_ATOMS_IN_MOL[dataset], scaler=scaler,
        use_layer_norm=False, schnet_hidden_channels=schnet_hidden_channels,
        schnet_num_filters=schnet_num_filters, schnet_num_interactions=schnet_num_interactions,
        set_transformer_hidden_dim=set_transformer_hidden_dim, set_transformer_num_heads=set_transformer_num_heads,
        set_transformer_num_sabs=set_transformer_num_sabs, lbl_or_emb=lbl_or_emb,
        include=include, emb_type=emb_type, property='homo' if target_id == 4 else 'lumo',
        aux_dim=aux_dim if aux_dim is not None else 0
    )

    model = Estimator(**gnn_args)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_loss',
        dirpath=OUT_DIR,
        filename='{epoch:04d}',
        mode='min',
        save_top_k=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_total_loss',
        patience=300,
        mode='min'
    )

    logger = WandbLogger(
        name=NAME,
        project='Multi-fidelity SchNet'
    )

    common_trainer_args = dict(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        num_sanity_val_steps=0,
        devices=1,
        min_epochs=1,
        max_epochs=-1
    )

    if argsdict['use_cuda']:
        model = model.cuda()
        trainer_args = common_trainer_args | dict(accelerator='gpu')
    else:
        trainer_args = common_trainer_args | dict(accelerator='cpu')


    if argsdict['ckpt_path']:
        trainer_args = dict(resume_from_checkpoint=argsdict['ckpt_path']) | trainer_args

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)

    # Save test metrics
    np.save(os.path.join(OUT_DIR, 'test_y_pred.npy'), model.test_output)
    np.save(os.path.join(OUT_DIR, 'test_y_true.npy'), model.test_true)
    np.save(os.path.join(OUT_DIR, 'test_metrics.npy'), model.test_metrics)
    
    # Uncomment if you want to save the test graph embeddings
    # np.save(os.path.join(OUT_DIR, 'test_graph_embeddings.npy'), model.test_graph_embeddings)


if __name__ == "__main__":
    main()
