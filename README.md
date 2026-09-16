# HybridCNN-RS: A Multi-Dataset Benchmark for Remote Sensing Scene Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Research-In%20Progress-yellow)
![Accuracy](https://img.shields.io/badge/Best%20Acc-95.24%25-brightgreen)

**HybridCNN-RS** is a lightweight, attention-augmented Convolutional Neural Network designed for high-performance Remote Sensing Scene Classification (RSSC). The model integrates multiple state-of-the-art attention mechanisms and multi-scale feature extraction modules into a single, computationally efficient backbone, and is benchmarked across five standard RSSC datasets.

## 📌 Highlights

- **94.62% test accuracy** on NWPU-RESISC45 (45 classes) with only **14.17M parameters** and **3.33 GFLOPs** — competitive with far heavier ResNet/ViT baselines.
- **First-principles ablation study** isolating the contribution of each module (SE, CBAM, Self-Attention, ASPP) across **5 datasets**.
- **5-Fold Stratified Cross-Validation** with statistical reporting (mean ± std).
- **Comprehensive evaluation suite:** confusion matrices, ROC/PR curves, calibration curves, misclassification analysis, and model complexity benchmarking (FLOPs, inference time).
- **Automatic checkpoint resumption** for long-running ablation sweeps — training survives Colab/Kaggle disconnections.

## 🏗️ Architecture

The model is a hybrid CNN built from five composable modules, each of which can be toggled on/off for ablation:

| Module | Purpose | Location |
|--------|---------|----------|
| **SE Blocks** | Channel-wise recalibration (squeeze-excitation) | Stages 2 & 3 |
| **CBAM** | Combined channel + spatial attention | Stage 3 |
| **Lightweight Self-Attention** | Multi-head self-attention over spatial positions | After Stage 3 |
| **ASPP** | Multi-scale dilated convolution for varying object sizes | After Stage 2 |
| **Stochastic Depth** | Linearly-scheduled dropout on residual branches | All stages |


svgsvg

Input (256×256×3)
↓
Stem (Conv → BN → ReLU → Conv → BN → ReLU → MaxPool)
↓
Stage 1 (2× ResidualBlock)
↓
Stage 2 (3× ResidualBlock + SE)
↓
ASPP (dilated convs, rates 1 & 3) ← optional
↓
Stage 3 (4× ResidualBlock + SE + CBAM)
↓
Lightweight Multi-Head Self-Attention ← optional
↓
Stage 4 (2× ResidualBlock)
↓
Global Avg Pool → Dropout → FC(256) → FC(num_classes)

text

### Training Recipe
- **Optimizer:** AdamW with decoupled weight decay (0.01), β = (0.9, 0.999)
- **Scheduler:** Warmup (5 epochs) + Cosine Annealing with restarts at epochs 100 & 150
- **Loss:** Label Smoothing Cross-Entropy (ε = 0.1)
- **Augmentation:** RandomResizedCrop, H/V Flip, Rotation, Affine, ColorJitter, RandomErasing
- **MixUp (α=0.2) + CutMix (α=0.2)** applied stochastically
- **Mixed Precision (AMP)** training via `torch.cuda.amp`
- **Gradient clipping** at norm 1.0
- **Early stopping** with patience = 20

## 🧪 Benchmark Results

### Main Result — NWPU-RESISC45 (45 classes)

| Metric | Value |
|--------|-------|
| Test Accuracy | **94.62%** |
| Precision (weighted) | 94.73% |
| Recall (weighted) | 94.62% |
| F1-Score (weighted) | 94.61% |
| Cohen's Kappa | 94.50% |
| Balanced Accuracy | 94.60% |
| MCC | 94.50% |
| Specificity | 99.88% |
| High-Confidence Failures | **0.00%** |
| Trainable Parameters | 14,176,501 |
| FLOPs | 3.33 G |
| Inference Time | ~X ms/image (RTX A4500) |

### Datasets Benchmarked

| Dataset | Classes | Train / Test | Notes |
|---------|---------|--------------|-------|
| NWPU-RESISC45 | 45 | 27,000 / 4,500 | Primary benchmark |
| EuroSAT | 10 | — | Land-use classification |
| PatternNet | 38 | — | Google Earth imagery |
| MLRSNet | 46 | — | Multi-label RS |
| AID | 30 | — | Aerial scene dataset |

*(Ablation results across all 5 datasets are stored in `ablation_results.csv`. See Cell 9 for the automated runner.)*

## ⚙️ Tech Stack

- **Deep Learning:** PyTorch, torchvision, AMP
- **Evaluation:** scikit-learn, seaborn, matplotlib
- **Complexity Profiling:** `thop` (FLOPs), custom inference-time benchmark
- **Logging:** TensorBoard (`SummaryWriter`)
- **Data Handling:** PIL, NumPy, Pandas

## 🚀 Installation & Usage

bash
# Clone the repository
git clone https://github.com/yourusername/HybridCNN-RS.git
cd HybridCNN-RS

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
pip install tqdm tensorboard thop

svgsvg

Data Preparation

Your dataset directory must follow this structure:

text

Datasets_Prepared/
└── NWPU/
    ├── train/
    │   ├── airplane/
    │   ├── baseball_diamond/
    │   └── ... (45 class folders)
    └── test/
        ├── airplane/
        └── ...

svgsvg

Edit the dataset_roots dictionary in the Config class to point to your local paths.

Running the Full Pipeline

Open New.ipynb and run cells sequentially:

Cell 1–7: Define config, dataset class, augmentation, model, loss, and trainer.

Cell 8: Train the full model on NWPU-RESISC45 and print final test metrics.

Cell 9: Run the automated ablation study across all 5 datasets. Skips completed experiments and resumes from checkpoints.

Cell 10: Run 5-Fold Stratified CV on a chosen dataset (e.g., EuroSAT).

Cell 11: Final evaluation of the "Full Model" on every dataset.

📂 Repository Structure

text

HybridCNN-RS/
├── data/                          # (not committed) — link your local datasets here
├── notebooks/
│   └── New.ipynb                  # Full training + ablation + CV pipeline
├── checkpoints/                   # best_model.pth, per-experiment folders
├── results/
│   ├── ablation_results.csv
│   ├── class_wise_metrics.csv
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── calibration_curve.png
│   └── confused_pairs.png
├── requirements.txt
└── README.md

svgsvg

📊 Generated Artifacts

The evaluation pipeline automatically produces publication-ready figures at 600 DPI:

Confusion Matrix (row-normalized, dynamic size for 10–46 classes)

Macro-Average ROC Curves for all classes

Per-Class Precision-Recall Curves with AP scores

Calibration Curve (reliability diagram)

Top-10 Most Misclassified Classes (horizontal bar chart)

Top-5 Most Confused Class Pairs

Training Curves (loss + accuracy, train vs. val)

Class-Wise Metrics CSV (accuracy, precision, recall, F1)

Complexity Report (params, FLOPs, inference time)

🔬 Key Design Decisions

Avoiding double attention. SE is disabled in Stage 3 when CBAM is active, since CBAM already contains a channel-attention path. This prevents redundant recalibration and is exposed as an ablation config.

Lightweight self-attention. Uses gamma-scaled residual attention (à la SAGAN) initialized to zero, so the module starts as identity and gradually learns to contribute — critical for stable training.

ASPP with only 2 rates. Reduces parameters and overfitting risk for RSSC tasks where receptive-field diversity matters more than depth.

Progressive Stochastic Depth. Drop-path probability increases linearly across the 16 residual blocks (drop_path_rate * block_idx / (total - 1)), matching the regularisation schedule of modern CNNs.

Warmup + Cosine with Restarts. Restarts at epochs 100 and 150 help escape shallow local minima observed in the 150-epoch schedule.


🔬 Future Work

Publish the ablation heatmap as an interactive table across all 5 datasets.

Compare against Vision Transformer baselines (ViT-B/16, Swin-T).

Add Grad-CAM visualisations to interpret which regions drive predictions.

Investigate domain adaptation from NWPU → AID for cross-dataset transfer.

👤 Author

Muhammad Ahmad

BS Artificial Intelligence Student

HITEC University, Pakistan

https://linkedin.com/in/ahmadhere

ahmadupworking@gmail.com

📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

🙏 Acknowledgements

NWPU-RESISC45 (Cheng et al., IEEE JSTARS 2017)

EuroSAT (Helber et al., IEEE GRSL 2019)

PatternNet (Zhou et al., 2018)

MLRSNet (Qi et al., IEEE JSTARS 2020)

AID (Xia et al., IEEE GRSL 2017)
