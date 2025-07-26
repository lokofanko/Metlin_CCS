import os
import argparse
import yaml
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pandas as pd

from data_processing import load_and_preprocess_data
from model import GCNWithCategoricalFeature
from train import _prepare_dataloader

def calculate_metrics(y_true, y_pred):
    """Рассчитывает набор метрик для оценки модели."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mdae = np.median(np.abs(y_true - y_pred))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "MDAE": mdae
    }

def evaluate_on_dataset(model, loader, device, dataset_name):
    """Проводит оценку на одном конкретном датасете и возвращает метрики."""
    model.eval()
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on {dataset_name}"):
            graph_data, cat_features = batch
            graph_data, cat_features = graph_data.to(device), cat_features.to(device)
            
            output = model([graph_data, cat_features])
            
            all_preds.extend(output.cpu().numpy().flatten())
            all_true.extend(graph_data.y.cpu().numpy().flatten())
            
    return calculate_metrics(all_true, all_preds)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained GNN model on test sets.')
    parser.add_argument('--experiment_dir', type=str, required=True, 
                        help='Path to the experiment directory containing model and config.')
    args = parser.parse_args()

    # --- 1. Загрузка конфигурации ---
    config_path = os.path.join(args.experiment_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_path = os.path.join(args.experiment_dir, 'best_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Загрузка данных ---
    data_paths = {k: os.path.join(config['data']['dir'], v) for k, v in config['data'].items() if k != 'dir'}
    # Загружаем только тестовые наборы
    processed_data = load_and_preprocess_data(data_paths, load_train=False, load_valid=False, load_tests=True)
    test1_data = processed_data['test1_data']
    test2_data = processed_data['test2_data']

    # --- 3. Создание DataLoader'ов ---
    test1_loader = _prepare_dataloader(test1_data, config['training']['batch_size'], shuffle=False)
    test2_loader = _prepare_dataloader(test2_data, config['training']['batch_size'], shuffle=False)
    
    # --- 4. Загрузка модели ---
    model_cfg = config['model']
    model = GCNWithCategoricalFeature(
        num_graph_features=model_cfg['num_graph_features'],
        num_cat_features=model_cfg['num_cat_features'],
        hidden_dim=model_cfg['hidden_dim'],
        fc_hidden_dim=model_cfg['fc_hidden_dim']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    # --- 5. Оценка и вывод результатов ---
    print("\n--- Final Evaluation Results ---")
    
    metrics_test1 = evaluate_on_dataset(model, test1_loader, device, "Test Set 1 (Xie et al.)")
    print(f"\nMetrics for Test Set 1 (Xie et al.):")
    for name, value in metrics_test1.items():
        print(f"  {name}: {value:.4f}")
        
    metrics_test2 = evaluate_on_dataset(model, test2_loader, device, "Test Set 2 (External)")
    print(f"\nMetrics for Test Set 2 (External):")
    for name, value in metrics_test2.items():
        print(f"  {name}: {value:.4f}")
        
    print("\n--------------------------------")

if __name__ == '__main__':
    main()