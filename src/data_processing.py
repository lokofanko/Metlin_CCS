import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# =============================================================================
# Класс для создания графов
# =============================================================================

class MolecularGraph:
    def __init__(self, use_chirality=True, hydrogens_implicit=True, use_stereochemistry=True):
        self.use_chirality = use_chirality
        self.hydrogens_implicit = hydrogens_implicit
        self.use_stereochemistry = use_stereochemistry

    def _one_hot_encoding(self, x, permitted_list):
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(x == s) for s in permitted_list]
        return binary_encoding

    def get_atom_features(self, atom):
        permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        
        atom_feature_vector = \
            self._one_hot_encoding(atom.GetSymbol(), permitted_list_of_atoms) + \
            self._one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, "MoreThanFour"]) + \
            self._one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]) + \
            self._one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]) + \
            [int(atom.IsInRing()), int(atom.GetIsAromatic())] + \
            [float((atom.GetMass() - 10.812) / 116.092), float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6), float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

        if self.use_chirality:
            atom_feature_vector += self._one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        if self.hydrogens_implicit:
            atom_feature_vector += self._one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "MoreThanFour"])
        return np.array(atom_feature_vector)

    def get_bond_features(self, bond):
        permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_feature_vector = self._one_hot_encoding(bond.GetBondType(), permitted_bond_types) + \
                              [int(bond.GetIsConjugated()), int(bond.IsInRing())]
        if self.use_stereochemistry:
            bond_feature_vector += self._one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        return np.array(bond_feature_vector)
    
    def smiles_to_graph_list(self, smiles_list, labels):
        data_list = []
        for smiles, label in zip(smiles_list, labels):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: continue

            atom_features_np = np.array([self.get_atom_features(atom) for atom in mol.GetAtoms()])
            atom_features = torch.tensor(atom_features_np, dtype=torch.float)
            
            adj = GetAdjacencyMatrix(mol)
            rows, cols = np.nonzero(adj)
            edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
            
            edge_features_list = [self.get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j))) for i, j in zip(rows, cols)]
            if edge_features_list:
                edge_attr_np = np.array(edge_features_list)
                edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
            else: 
                edge_attr = torch.empty((0, 10), dtype=torch.float)

            y = torch.tensor([label], dtype=torch.float)
            data_list.append(Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, y=y))
        return data_list

# =============================================================================
# Вспомогательные функции
# =============================================================================

def _filter_invalid_smiles(df, smiles_column='smiles'):
    """Удаляет строки с невалидными SMILES."""
    is_valid = df[smiles_column].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    return df[is_valid].reset_index(drop=True)

def _one_hot_encode(df, feature_column='Adduct'):
    """Создает one-hot кодирование для указанной колонки."""
    return pd.get_dummies(df[feature_column], prefix=feature_column, dtype=float)

# =============================================================================
# Главная функция загрузки и обработки данных
# =============================================================================

def load_and_preprocess_data(data_paths, load_train=True, load_valid=True, load_tests=True):
    """
    Основная функция для загрузки и полной предобработки данных.
    
    Args:
        data_paths (dict): Словарь с путями к файлам.
        load_train (bool): Загружать ли обучающий набор.
        load_valid (bool): Загружать ли валидационный набор.
        load_tests (bool): Загружать ли тестовые наборы.
    
    Returns:
        dict: Словарь с загруженными данными.
    """
    print("Loading and preprocessing data...")
    graph_creator = MolecularGraph()
    
    def _process_df(df):
        df_valid = _filter_invalid_smiles(df)
        smiles = df_valid.smiles.to_numpy()
        ccs_values = df_valid.CCS_AVG.to_numpy()
        graphs = graph_creator.smiles_to_graph_list(smiles, ccs_values)
        cat_features = _one_hot_encode(df_valid, 'Adduct')
        return [graphs, cat_features]

    results = {}

    if load_train or load_valid:
        print("Loading train/validation sets...")
        data_main = pd.read_csv(data_paths['main']).drop(columns='Unnamed: 0', errors='ignore')
        train_main_df, valid_main_df = train_test_split(data_main, test_size=0.1, random_state=42)
        
        if load_valid:
            results['valid_data'] = _process_df(valid_main_df)
        
        if load_train:
            data_external_train = pd.read_csv(data_paths['external_train'])
            train_df = pd.concat([train_main_df, data_external_train]).reset_index(drop=True)
            
            print("Processing Tanimoto weights...")
            tan_df = pd.read_csv(data_paths['tanimoto'])
            if "Unnamed: 0" in tan_df.columns:
                tan_df = tan_df.drop(columns=["Unnamed: 0"])
            
            train_df = pd.merge(train_df, tan_df[['smiles', 'mean_top_a_tanimoto']], on='smiles', how='left')
            # Исправляем FutureWarning от Pandas
            train_df['mean_top_a_tanimoto'] = train_df['mean_top_a_tanimoto'].fillna(0)
            
            results['train_data'] = _process_df(train_df)
            results['tanimoto_weights'] = (1 - train_df['mean_top_a_tanimoto']).to_numpy()

    if load_tests:
        print("Loading test sets...")
        test1_df = pd.read_csv(data_paths['test1'])
        test2_df = pd.read_csv(data_paths['test2'])
        results['test1_data'] = _process_df(test1_df)
        results['test2_data'] = _process_df(test2_df)
        
    print("Data processing complete.")
    return results