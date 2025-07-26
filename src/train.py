import os
import argparse
import yaml
import shutil
import statistics
import torch
from torch import nn
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_processing import load_and_preprocess_data
from model import GCNWithCategoricalFeature

def _prepare_dataloader(data, batch_size, shuffle, use_weights=False, weights=None):
    """Вспомогательная функция для создания DataLoader."""
    graphs, cat_features = data
    cat_features_tensor = torch.tensor(cat_features.values, dtype=torch.float)
    
    if use_weights and weights is not None:
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        dataset = list(zip(graphs, cat_features_tensor, weights_tensor))
    else:
        dataset = list(zip(graphs, cat_features_tensor))
        
    def collate_fn(batch):
        graphs_batch = [item[0] for item in batch]
        cats_batch = torch.stack([item[1] for item in batch], dim=0)
        batched_graph = Batch.from_data_list(graphs_batch)
        
        if use_weights and weights is not None:
            weights_batch = torch.stack([item[2] for item in batch], dim=0)
            return batched_graph, cats_batch, weights_batch
        return batched_graph, cats_batch

    # num_workers=0 важен для Windows, чтобы избежать проблем с многопоточностью
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)


def validate(model, loader, loss_fn, device):
    """Оценивает модель на валидационном наборе."""
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch in loader:
            graph_data, cat_features = batch
            graph_data, cat_features = graph_data.to(device), cat_features.to(device)
            output = model([graph_data, cat_features])
            y_true = graph_data.y.view_as(output)
            loss = loss_fn(output, y_true).mean()
            total_loss.append(loss.item())
    return statistics.mean(total_loss)

def train(config):
    """Основной цикл обучения модели."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Создание директорий для вывода
    output_dir = os.path.join(config['training']['output_dir'], config['experiment_name'])
    log_dir = os.path.join(config['logging']['log_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Инициализация TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # 3. Загрузка данных
    data_paths = {k: os.path.join(config['data']['dir'], v) for k, v in config['data'].items() if k != 'dir'}
    processed_data = load_and_preprocess_data(data_paths, load_tests=False, load_train=True, load_valid=True)
    train_data = processed_data['train_data']
    valid_data = processed_data['valid_data']
    tanimoto_weights = processed_data.get('tanimoto_weights')

    # 4. Создание DataLoader'ов
    train_loader = _prepare_dataloader(
        train_data, config['training']['batch_size'], shuffle=True, 
        use_weights=config['training']['use_tanimoto'], weights=tanimoto_weights
    )
    valid_loader = _prepare_dataloader(valid_data, config['training']['batch_size'], shuffle=False)
    
    # 5. Инициализация модели, оптимизатора и функции потерь
    model_cfg = config['model']
    model = GCNWithCategoricalFeature(
        num_graph_features=train_data[0][0].x.shape[1],
        num_cat_features=train_data[1].shape[1],
        hidden_dim=model_cfg['hidden_dim'],
        fc_hidden_dim=model_cfg['fc_hidden_dim']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    loss_fn = nn.L1Loss(reduction='none') # reduction='none' для возможности взвешивания

    # 6. Цикл обучения
    best_val_loss = float('inf')
    print("\nStarting training...")
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            use_tanimoto = config['training']['use_tanimoto']
            
            graph_data, cat_features, *weights_batch = batch
            weights = weights_batch[0] if use_tanimoto and weights_batch else None

            graph_data, cat_features = graph_data.to(device), cat_features.to(device)
            output = model([graph_data, cat_features])
            
            y_true = graph_data.y.view_as(output)
            loss_per_item = loss_fn(output, y_true)
            
            if use_tanimoto and weights is not None:
                loss = (loss_per_item * weights.to(device).unsqueeze(1)).mean()
            else:
                loss = loss_per_item.mean()
                
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            progress_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = statistics.mean(epoch_loss)
        avg_val_loss = validate(model, valid_loader, loss_fn, device)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1}: Train MAE = {avg_train_loss:.4f}, Val MAE = {avg_val_loss:.4f}")

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"-> Best model saved to {model_path} (Val MAE: {best_val_loss:.4f})")
            
        # Сохранение чекпоинта каждые 5 эпох и на последней эпохе
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config['training']['epochs']:
            checkpoint_path = os.path.join(output_dir, 'checkpoint.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"-> Checkpoint saved to {checkpoint_path} at epoch {epoch+1}")


    writer.close()
    shutil.copy(args.config, os.path.join(output_dir, 'config.yaml'))
    print(f"\nTraining finished. Best validation MAE: {best_val_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN for CCS Prediction from a config file.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    exp_name = f"lr{config['training']['lr']}_bs{config['training']['batch_size']}"
    exp_name += "_with_tanimoto" if config['training']['use_tanimoto'] else "_no_weights"
    config['experiment_name'] = exp_name

    train(config)