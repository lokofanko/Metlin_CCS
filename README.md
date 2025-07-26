
# \# High-Accuracy CCS Prediction using Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-green.svg)](https://www.rdkit.org/)

This repository contains the official implementation of a high-accuracy and efficient Graph Neural Network (GNN) model for predicting Collision Cross-Section (CCS) values in ion mobility spectrometry.

## 📜 About The Project

Ion Mobility Spectrometry (IMS) is a powerful analytical technique for separating ions based on their size and shape. A key parameter in IMS is the **Collision Cross-Section (CCS)**. Accurate prediction of CCS values from molecular structures is crucial for identifying small molecules in complex mixtures, especially in non-target screening.

This project introduces a robust GNN model that predicts CCS values for both positive and negative ions based solely on their SMILES representation. Our approach utilizes an innovative weighting scheme based on **Tanimoto similarity**, which improves the model's accuracy by increasing the importance of structurally unique molecules during training.

### ✨ Key Features

- **State-of-the-Art Accuracy:** Achieves a Mean Absolute Error (MAE) of **3.57 Å²** on the benchmark test set from Xie et al. (2024), demonstrating superior performance.
- **Graph Neural Network (GNN):** Represents molecules as graphs to effectively capture topological and chemical features, leading to more accurate predictions.
- **Tanimoto Similarity Weighting:** Employs a sample weighting mechanism where the weight is calculated as `1 - Sa` (Sa being the average Tanimoto similarity to the `Ns=25` nearest neighbors). This improves generalization for underrepresented molecules.
- **Efficient Architecture:** Provides a balance between high accuracy and computational efficiency compared to deeper models.
- **Reproducible \& Configurable:** The entire pipeline is managed via a `config.yaml` file, and results can be tracked using TensorBoard.


## 📊 Results

The model was trained on a combined dataset from "METLIN-CCS" and Xie et al. and evaluated on two independent test sets.

### Performance Metrics

The table below shows the comprehensive performance of our model on the two test sets.


| Test Set | MAE (Å²) | RMSE (Å²) | MAPE (%) | MDAE (Å²) |
| :-- | :--: | :--: | :--: | :--: |
| **Test Set 1 (Xie et al.)** | **3.5658** | 4.9486 | 1.9854 | 2.5507 |
| **Test Set 2 (External)** | 10.0921 | 12.0486 | 3.7641 | 9.8741 |

### Comparison with State-of-the-Art

Our model shows a significant improvement in MAE compared to the previous state-of-the-art model by **Xie T. et al. (2024)** on the same test sets.


| Model | Test Set 1 MAE (Å²) | Test Set 2 MAE (Å²) |
| :-- | :--: | :--: |
| **Our Model (GNN + Tanimoto Weights)** | **3.57** | **10.09** |
| Xie T. et al. (2024) | *впиши сюда* | *впиши сюда* |

## ⚙️ Setup and Installation

1. **Clone the repository:**
    ```bash     git clone https://github.com/your-username/Metlin_CCS.git     cd Metlin_CCS     ```
2. **Create and activate a virtual environment (recommended):**
    ```bash     python -m venv venv     source venv/bin/activate  # On Windows: venv\Scripts\activate     ```
3. **Install dependencies:**
    ```bash     pip install -r requirements.txt     ```

## 🚀 Usage

### 1. Data Preparation

The raw data files (`.csv`) are not included in this repository due to their size. You can download the required dataset archive from the following link:

**[>> Download Dataset (Metlin_CCS_Data.zip) <<](https://drive.google.com/drive/folders/15gv2jS6eH-BfBCEpYVhYxKiJdO1gwcS9?usp=sharing)**

After downloading, unzip the archive. Place all `.csv` files inside the `data/raw/` directory. The final structure should look like this:

Metlin_CCS/
└── data/
└── raw/
├── data_main_stand.csv
├── data_external_stand.csv
├── external_test_set1_stand.csv
├── external_test_set2_stand.csv
└── tanimoto_spec.csv

The project is now ready for training. No further preprocessing is needed as the scripts handle it automatically.

### 2. (Optional) Generating Tanimoto Weights

The `tanimoto_spec.csv` file, used for sample weighting, is provided in the dataset archive. However, you can regenerate it using the following script. This is useful if you use a different dataset or change the number of nearest neighbors (`Ns`).

For our experiments, we used `Ns=25`.

```bash
python src/generate_tanimoto.py --main_data_path data/raw/data_main_stand.csv --external_data_path data/raw/data_external_stand.csv --a 25


3. Training a New Model
All training parameters are controlled via config.yaml. To start training, run: python src/train.py --config config.yaml


Training logs will be saved in the runs/ directory, and the best model will be saved in models/. You can monitor the training process using TensorBoard: tensorboard --logdir runs


4. Evaluating the Trained Model
To evaluate the best model from a training run on the test sets, use the evaluate.py script. You need to provide the path to the specific experiment directory.
python src/evaluate.py --experiment_dir models/lr0.0001_bs32_with_tanimoto


5. Predicting for a Single Molecule
Use the predict.py script to get a CCS prediction for a new molecule.
python src/predict.py --model_path models/lr0.0001_bs32_with_tanimoto/best_model.pt --smiles "CCO" --adduct "[M+H]+"


📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
✍️ Citation
If you use this code or our findings in your research, please consider citing this repository.
[Your Name/Group]. (Year). High-Accuracy CCS Prediction using Graph Neural Networks. GitHub. https://github.com/your-username/Metlin_CCS

ты можешь из этого сделать нормальный ридми файл ну или сайт куда я могу это вставить и оно отредактируется

Of course. Here is the provided text formatted into a clean and professional `README.md` file. You can copy and paste this directly into a `README.md` file on GitHub or another platform that supports Markdown.

# High-Accuracy CCS Prediction using Graph Neural Networks

[[^1][^2]
[[^3]
[
[

This repository contains the official implementation of a high-accuracy and efficient Graph Neural Network (GNN) model for predicting Collision Cross-Section (CCS) values in ion mobility spectrometry.

## 📜 About The Project

Ion Mobility Spectrometry (IMS) is a powerful analytical technique for separating ions based on their size and shape. A key parameter in IMS is the **Collision Cross-Section (CCS)**. Accurate prediction of CCS values from molecular structures is crucial for identifying small molecules in complex mixtures, especially in non-target screening.

This project introduces a robust GNN model that predicts CCS values for both positive and negative ions based solely on their SMILES representation. Our approach utilizes an innovative weighting scheme based on **Tanimoto similarity**, which improves the model's accuracy by increasing the importance of structurally unique molecules during training.

### ✨ Key Features

*   **State-of-the-Art Accuracy:** Achieves a Mean Absolute Error (MAE) of **3.57 Å²** on the benchmark test set from Xie et al. (2024), demonstrating superior performance.
*   **Graph Neural Network (GNN):** Represents molecules as graphs to effectively capture topological and chemical features, leading to more accurate predictions.
*   **Tanimoto Similarity Weighting:** Employs a sample weighting mechanism where the weight is calculated as `1 - Sa` (Sa being the average Tanimoto similarity to the `Ns=25` nearest neighbors). This improves generalization for underrepresented molecules.
*   **Efficient Architecture:** Provides a balance between high accuracy and computational efficiency compared to deeper models.
*   **Reproducible & Configurable:** The entire pipeline is managed via a `config.yaml` file, and results can be tracked using TensorBoard.

## 📊 Results

The model was trained on a combined dataset from "METLIN-CCS" and Xie et al. and evaluated on two independent test sets.

### Performance Metrics

The table below shows the comprehensive performance of our model on the two test sets.

| Test Set                  | MAE (Å²) | RMSE (Å²) | MAPE (%) | MDAE (Å²) |
| :------------------------ | :------: | :-------: | :------: | :-------: |
| **Test Set 1 (Xie et al.)** | **3.5658** |  4.9486   |  1.9854  |  2.5507   |
| **Test Set 2 (External)** | 10.0921  |  12.0486  |  3.7641  |  9.8741   |

### Comparison with State-of-the-Art

Our model shows a significant improvement in MAE compared to the previous state-of-the-art model by **Xie T. et al. (2024)** on the same test sets.

| Model                                | Test Set 1 MAE (Å²) | Test Set 2 MAE (Å²) |
| :----------------------------------- | :-----------------: | :-----------------: |
| **Our Model (GNN + Tanimoto Weights)** |      **3.57**       |      **10.09**      |
| Xie T. et al. (2024)                 |  **_FILL IN VALUE_**  |  **_FILL IN VALUE_**  |

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Metlin_CCS.git
    cd Metlin_CCS
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Data Preparation

The raw data files (`.csv`) are not included in this repository due to their size. You can download the required dataset archive from the following link:

**[>> Download Dataset (Metlin\_CCS\_Data.zip) <<](https://drive.google.com/drive/folders/15gv2jS6eH-BfBCEpYVhYxKiJdO1gwcS9?usp=sharing)**

After downloading, unzip the archive. Place all `.csv` files inside the `data/raw/` directory. The final structure should look like this:

```

Metlin_CCS/
└── data/
└── raw/
├── data_main_stand.csv
├── data_external_stand.csv
├── external_test_set1_stand.csv
├── external_test_set2_stand.csv
└── tanimoto_spec.csv

```

The project is now ready for training. No further preprocessing is needed as the scripts handle it automatically.

### 2. (Optional) Generating Tanimoto Weights

The `tanimoto_spec.csv` file, used for sample weighting, is provided in the dataset archive. However, you can regenerate it using the following script. This is useful if you use a different dataset or change the number of nearest neighbors (`Ns`).

For our experiments, we used `Ns=25`.
```bash
python src/generate_tanimoto.py --main_data_path data/raw/data_main_stand.csv --external_data_path data/raw/data_external_stand.csv --a 25
```


### 3. Training a New Model

All training parameters are controlled via `config.yaml`. To start training, run:

```bash
python src/train.py --config config.yaml
```

Training logs will be saved in the `runs/` directory, and the best model will be saved in `models/`. You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir runs
```


### 4. Evaluating the Trained Model

To evaluate the best model from a training run on the test sets, use the `evaluate.py` script. You need to provide the path to the specific experiment directory.

```bash
python src/evaluate.py --experiment_dir models/lr0.0001_bs32_with_tanimoto
```


### 5. Predicting for a Single Molecule

Use the `predict.py` script to get a CCS prediction for a new molecule.

```bash
python src/predict.py --model_path models/lr0.0001_bs32_with_tanimoto/best_model.pt --smiles "CCO" --adduct "[M+H]+"
```


## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## ✍️ Citation

If you use this code or our findings in your research, please consider citing this repository.

```
[Your Name/Group]. (Year). High-Accuracy CCS Prediction using Graph Neural Networks. GitHub. https://github.com/your-username/Metlin_CCS
```

<div style="text-align: center">⁂</div>

[^1]: https://img.shields.io/badge/License-MIT-yellow.svg

[^2]: https://opensource.org/licenses/MIT

[^3]: https://img.shields.io/badge/python-3.9+-blue.svg

