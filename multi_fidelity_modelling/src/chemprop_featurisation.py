from rdkit import Chem
from typing import Sequence, Dict


def onek_encoding_unk(value: int, choices: Sequence):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding
    

def atom_features(atom: Chem.rdchem.Atom, features_constants: Dict[str, Sequence], functional_groups=None):
    features = onek_encoding_unk(atom.GetAtomicNum(), features_constants['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), features_constants['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), features_constants['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), features_constants['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), features_constants['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), features_constants['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def get_atom_constants(max_atomic_num: int):
    return {
    'atomic_num': list(range(max_atomic_num)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}