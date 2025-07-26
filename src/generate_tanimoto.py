import argparse
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

def calculate_fingerprints(smiles_list):
    """Вычисляет фингерпринты Morgan (ECFP4) для списка SMILES."""
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fingerprints.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(fp)
    return fingerprints

def main(args):
    """
    Основная функция для вычисления среднего сходства Танимото 
    и сохранения результатов в CSV файл.
    """
    print("Loading datasets...")
    # Загружаем все данные, которые используются для обучения, чтобы посчитать Tanimoto на полном наборе
    data_main = pd.read_csv(args.main_data_path).drop(columns='Unnamed: 0', errors='ignore')
    data_external = pd.read_csv(args.external_data_path)
    full_train_df = pd.concat([data_main, data_external]).reset_index(drop=True)

    smiles_list = full_train_df[args.smiles_column].tolist()

    print("Calculating fingerprints...")
    fingerprints = calculate_fingerprints(smiles_list)
    
    valid_fingerprints = [fp for fp in fingerprints if fp is not None]
    
    mean_similarities = []
    print(f"Calculating mean Tanimoto similarity for top {args.a} neighbors...")
    
    for i, fp1 in enumerate(tqdm(fingerprints)):
        if fp1 is None:
            mean_similarities.append(0.0)
            continue
            
        # Вычисляем сходство со всеми остальными валидными фингерпринтами
        similarities = DataStructs.BulkTanimotoSimilarity(fp1, valid_fingerprints)
        
        # Сортируем по убыванию. Первое значение будет 1.0 (сходство с самим собой), его нужно отбросить.
        similarities.sort(reverse=True)
        
        # Берем top-A соседей (не включая саму молекулу)
        top_a_similarities = similarities[1:args.a + 1]
        
        if top_a_similarities:
            mean_top_a = np.mean(top_a_similarities)
            mean_similarities.append(mean_top_a)
        else:
            mean_similarities.append(0.0)

    # Создаем DataFrame для сохранения
    result_df = pd.DataFrame({
        'smiles': smiles_list,
        'mean_top_a_tanimoto': mean_similarities,
        'Unnamed: 0': full_train_df.index # Сохраняем оригинальные индексы для сопоставления
    })

    result_df.to_csv(args.output_path, index=False)
    print(f"Tanimoto weights successfully saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Tanimoto similarity weights for a dataset.")
    parser.add_argument('--main_data_path', type=str, required=True, help='Path to the main training data CSV file.')
    parser.add_argument('--external_data_path', type=str, required=True, help='Path to the external training data CSV file.')
    parser.add_argument('--output_path', type=str, default='data/raw/tanimoto_spec.csv', help='Path to save the output CSV file.')
    parser.add_argument('--smiles_column', type=str, default='smiles', help='Name of the SMILES column.')
    parser.add_argument('--a', type=int, default=25, help='Number of top similar molecules to consider (Ns).')
    
    args = parser.parse_args()
    main(args)