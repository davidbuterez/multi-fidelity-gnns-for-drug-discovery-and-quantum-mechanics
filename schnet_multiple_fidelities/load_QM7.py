import torch
from torch_geometric.data import Data

from .loading_utils import train_scaler, scale_dataset_aux


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


def np_to_geometric_data(ds_array, target_id):
    geom_data = []
    for data in ds_array:
        data_obj = Data(
            formula=data.formula,
            z=torch.from_numpy(data.z).to(torch.long),
            pos=torch.from_numpy(data.pos).to(torch.float),
            y=torch.from_numpy(data.y.astype(float)).to(torch.float)[target_id].reshape(1,),
            y_names=[data.y_names[target_id]],
            num_nodes=data.z.shape[0],
            homo_zindo_sum_emb=torch.from_numpy(data.homo_zindo_sum_emb).to(torch.float),
            homo_zindo_st_emb=torch.from_numpy(data.homo_zindo_st_emb).to(torch.float),
            homo_pbe0_sum_emb=torch.from_numpy(data.homo_pbe0_sum_emb).to(torch.float),
            homo_pbe0_st_emb=torch.from_numpy(data.homo_pbe0_st_emb).to(torch.float),
            lumo_zindo_sum_emb=torch.from_numpy(data.lumo_zindo_sum_emb).to(torch.float),
            lumo_zindo_st_emb=torch.from_numpy(data.lumo_zindo_st_emb).to(torch.float),
            lumo_pbe0_sum_emb=torch.from_numpy(data.lumo_pbe0_sum_emb).to(torch.float),
            lumo_pbe0_st_emb=torch.from_numpy(data.lumo_pbe0_st_emb).to(torch.float),
            homo_zindo_label=torch.tensor(data.homo_zindo_label),
            homo_pbe0_label=torch.tensor(data.homo_pbe0_label),
            homo_gw_label=torch.tensor(data.homo_gw_label),
            lumo_zindo_label=torch.tensor(data.lumo_zindo_label),
            lumo_pbe0_label=torch.tensor(data.lumo_pbe0_label),
            lumo_gw_label=torch.tensor(data.lumo_gw_label),
        )
        geom_data.append(data_obj)

    return geom_data


def load_QM7_aux(random_seed:int, target_property_id: int):
    assert random_seed in [23887, 386333, 514094, 572909, 598587]
    assert target_property_id in range(14)

    ds = torch.load('/home/david/Projects_HTS_Quantum_Rev/quantum-3d-multifidelity-SCP/QM7b_aux.pt')

    train, val, test = torch.utils.data.random_split(ds, [5769, 721, 721], generator=torch.Generator().manual_seed(random_seed))

    train_geometric = np_to_geometric_data(train, target_property_id)
    val_geometric = np_to_geometric_data(val, target_property_id)
    test_geometric = np_to_geometric_data(test, target_property_id)

    scaler = train_scaler(train_geometric)

    train_scaled = scale_dataset_aux(train_geometric, scaler)
    val_scaled = scale_dataset_aux(val_geometric, scaler)
    test_scaled = scale_dataset_aux(test_geometric, scaler)

    return train_scaled, val_scaled, test_scaled, scaler
