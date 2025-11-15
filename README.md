# ✂️ iMAD Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference

## Abstract
Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accu- rate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test-dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we show that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).

<p align="center">
  <img src="Header.png" alt="Header" width="400">
</p>

## 🚀 Features

- 🔧 Fully configurable via CLI  
- 📊 Auto‑scaling, balancing (SMOTE / Undersampling)  
- 🧮 Custom FocusCalLoss  
- 🏋️‍♂️ Train/test metrics with ROC‑AUC  
- 🗂️ Automatic results logging + model saving  

## 📦 Installation

```bash
conda create -n imad python=3.10
conda activate imad
pip install -r requirements.txt
```

Python 3.12.6 is recommended.

## 📁 Project Structure

```
.
├── Classfier.py      # Main training script
├── DataLoader.py     # Data loading utilities
├── Model.py          # MLP2Head definition
├── Losses.py         # Custom loss functions
└── Results/          # CSV logs
```

## ▶️ Quick Start

### Run with defaults

```bash
python Classfier.py
```

### Example with full custom settings

```bash
python Classfier.py   --Model MLP2HEAD     --lossName FocusCalLoss   --Nlayers 6   --hidden_dim 200   --epochs 20   --Learningrate 0.001   --Sampler SMOTE   --Scaller standard   --ClassWeights True
```

## ⚙️ Command-Line Arguments

### 📌 Model & Dataset

| Argument | Default | Choices | Description |
|---------|---------|---------|-------------|
| --datapath | Dataset | - | Directory path of dataset |
| --Data | stat_self_critique | stat_self_critique | Dataset name |
| --confcolumn | InitialConfidence | - | Confidence column name |

### 🏋️ Training

| Argument | Default | Choices | Description |
|---------|---------|---------|-------------|
| --Nlayers | 5 | 2–5 | Number of MLP layers |
| --hidden_dim | 200 | 200/500/1000 | Hidden layer size |
| --dropout_rate | 0.2 | - | Dropout |
| --use_batchnorm | True | True/False | Use BatchNorm |
| --Learningrate | 0.001 | - | Learning rate |
| --epochs | 50 | - | Epochs |

### 🔄 Preprocessing

| Argument | Default | Choices | Description |
|---------|---------|---------|-------------|
| --Sampler | none | SMOTE/RandomUnderSampler/none | Imbalance handling |
| --Scaller | standard | standard/minmax/none | Feature scaling |
| --ClassWeights | False | True/False | Weighted loss |

### 🔥 FocusCalLoss Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| --alpha_pos | 2.0 | Positive weight |
| --alpha_neg | 1.0 | Negative weight |
| --gamma | 2 | Focusing parameter |
| --lambda_cp | 6 | Regularization term |
| --mu_ece | 5 | Calibration loss weight |
| --tau | 0.7 | Temperature scaling |
| --n_bins | 15 | ECE bins |


## 📤 Output

- **Models** saved in `DLModels/Model_{EXP}.pt`
- **Scalers** saved as `.joblib`
- **Results CSV** in `Results/DLLResults.csv`  
  Includes accuracy, precision, recall, F1, ROC‑AUC, confusion matrix.

