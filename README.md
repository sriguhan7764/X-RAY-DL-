# NIH Chest X-Ray Multi-Label Classification
### 24AI636 Deep Learning — Mini-Projects 1, 2, 3 & 4

Multi-label thoracic disease classification from the NIH ChestX-ray14 dataset, progressing from baseline MLP/CNN models through pretrained temporal architectures, generative modelling (Autoencoder & GAN), and a full production-ready end-to-end deep learning system.

---

## 📁 Project Structure

```
clinical-curator/
│
├── README.md                          # This file
├── CONTRIBUTING.md                    # Dev setup, code style, PR checklist
├── LICENSE                            # MIT
├── pyproject.toml                     # Modern Python packaging + ruff/mypy/pytest config
├── setup.py                           # pip install -e . support
├── environment.yml                    # Conda environment (Python 3.11, PyTorch 2.1)
├── docker-compose.yml                 # One-command full-stack launch
├── .gitignore
│
├── src/                               # Installable Python package
│   ├── data/
│   │   └── dataset.py                 # NIHChestXrayDataset, AdvancedXrayDataset,
│   │                                  # XrayDataset, TemporalPatientDataset,
│   │                                  # get_transforms, get_ae_transforms, get_dataloaders
│   ├── models/
│   │   ├── mlp.py                     # ChestMLP — 4-layer deep MLP (R1)
│   │   ├── cnn.py                     # ChestCNN, ConvBlock — 4-block CNN (R1)
│   │   ├── pretrained.py              # FeatureExtractor, FineTunedResNet-50 (R2)
│   │   ├── temporal.py                # CNN_RNN_Hybrid, BahdanauAttention,
│   │   │                              # SelfAttention, MultiHeadAttention (R2)
│   │   ├── autoencoder.py             # Encoder, Decoder, ConvAutoencoder, VAE,
│   │   │                              # vae_loss (R3)
│   │   ├── gan.py                     # Generator, Discriminator — DCGAN (R3)
│   │   └── densenet.py                # DenseNetCXR, EfficientNetCXR (R4)
│   ├── training/
│   │   └── trainer.py                 # Trainer — BCEWithLogitsLoss, cosine LR,
│   │                                  # early stopping, checkpoint saving
│   └── utils/
│       ├── gradcam.py                 # GradCAM — forward/backward hooks, overlay
│       ├── metrics.py                 # evaluate_model, per_class_auc, mean_auc
│       └── visualization.py           # plot_learning_curves, plot_roc_curves,
│                                      # plot_f1_heatmap
│
├── backend/                           # FastAPI inference server
│   ├── main.py                        # 7 REST endpoints: /health, /models,
│   │                                  # /predict, /predict/batch, /latent,
│   │                                  # /temporal, /report/pdf
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                          # React + TypeScript + Vite
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.tsx            # Hero, model showcase, sample runner
│   │   │   ├── Upload.tsx             # File drop, model multi-select
│   │   │   ├── Results.tsx            # GradCAM viewer, predictions, PDF/share
│   │   │   └── Compare.tsx            # 3-model card grid, ensemble summary
│   │   ├── components/ui/
│   │   │   ├── XrayViewer.tsx         # Image + GradCAM CSS overlay
│   │   │   ├── PredictionBar.tsx      # Confidence bar with risk colours
│   │   │   ├── ConfidenceBadge.tsx
│   │   │   ├── ModelCard.tsx
│   │   │   ├── UploadZone.tsx
│   │   │   └── LoadingSteps.tsx
│   │   ├── hooks/
│   │   │   ├── usePrediction.ts       # predict, predictBatch, status, error
│   │   │   └── useModels.ts           # fetch model list on mount
│   │   └── api/
│   │       └── client.ts              # Typed API client (all 7 endpoints)
│   └── Dockerfile
│
├── scripts/                           # CLI training entry points (argparse)
│   ├── train_r1.py                    # MLP + CNN with optional grid search
│   ├── train_r2.py                    # FineTunedResNet + CNN-RNN hybrid
│   ├── train_r3.py                    # AE + VAE + DCGAN with AMP
│   └── train_r4.py                    # DenseNet-121 with Optuna HPO + ablation
│
├── tests/                             # pytest test suite (no GPU or dataset needed)
│   ├── test_models.py                 # Forward-pass shape/value tests, frozen layers
│   ├── test_trainer.py                # Fit history, loss descent, early stopping
│   ├── test_api.py                    # TestClient: all endpoints, schema, error codes
│   └── test_gradcam.py                # Heatmap shape/range, overlay, argmax class
│
├── configs/models/                    # Per-model YAML hyperparameter configs
│   ├── mlp.yaml
│   ├── cnn.yaml
│   ├── pretrained.yaml
│   ├── temporal.yaml
│   ├── autoencoder.yaml
│   └── densenet.yaml
│
├── notebooks/                         # Original Kaggle notebooks (reference)
│   ├── review01_mlp_cnn.ipynb
│   ├── review02_pretrained_temporal.ipynb
│   ├── review03_ae_gan.ipynb
│   ├── review04_densenet_e2e.ipynb
│   └── master_combined.ipynb
│
└── sample_images/
    ├── 00016051_011.png               # NIH case PA view
    └── 00000003_000.png               # NIH case PA view
```

---

## 📊 Dataset

| Field | Detail |
|-------|--------|
| Name | NIH ChestX-ray14 |
| Total images | 112,120 frontal-view chest X-rays |
| Labels | 14 thoracic disease classes (multi-label) |
| Split | Patient-level 80% train / 10% val / 10% test |
| Input resolution | 224 × 224 × 3 (64 × 64 × 1 for AE/GAN) |
| Source | Downloaded via `kagglehub` |
| Seed | 42 (fully reproducible) |

**Disease Classes:**  
Atelectasis · Cardiomegaly · Consolidation · Edema · Effusion · Emphysema · Fibrosis · Hernia · Infiltration · Mass · Nodule · Pleural Thickening · Pneumonia · Pneumothorax

---

## 📓 Mini-Project 1 — MLP & CNN Classification

**Notebook:** `notebooks/review01_mlp_cnn.ipynb`  
**Review:** Review 1 — 12th Feb 2026

### Objectives
Implement and compare baseline deep learning models for multi-label chest X-ray classification, covering the full ML pipeline from preprocessing to explainability.

### Architectures

**MLP**
```
Input (150,528) → Dense(1024) → BN → ReLU → Dropout
               → Dense(512)  → BN → ReLU → Dropout
               → Dense(128)  → Dense(14) → Sigmoid
```

**CNN**
```
Input (3×224×224)
→ Conv Block 1: Conv2d(3→32)    + BN + ReLU + MaxPool(2×2)
→ Conv Block 2: Conv2d(32→64)   + BN + ReLU + MaxPool(2×2)
→ Conv Block 3: Conv2d(64→128)  + BN + ReLU + MaxPool(2×2)
→ Conv Block 4: Conv2d(128→256) + BN + ReLU + MaxPool(2×2)
→ FC(1024) → Dropout → FC(14) → Sigmoid
```

### Key Components

| Component | Choice | Reason |
|-----------|--------|--------|
| Loss | BCEWithLogitsLoss + pos_weight | Numerically stable; handles class imbalance |
| Optimizer | AdamW + weight decay | Adaptive LR with L2 regularisation |
| Scheduler | ReduceLROnPlateau | Adaptive LR based on val AUC |
| Augmentation | RandomHorizontalFlip + RandomRotation(10°) | Train split only |

### MLP vs CNN Summary

| Aspect | MLP | CNN |
|--------|-----|-----|
| Spatial awareness | ❌ Flattens image | ✅ Preserves local structure |
| Parameter efficiency | Low | High |
| Expected AUC | ~0.55–0.65 | ~0.65–0.75 |
| Training stability | May oscillate | Smoother convergence |

### Sections
1. Problem Formulation & Configuration
2. Data Preprocessing & Augmentation
3. MLP Implementation
4. CNN Implementation
5. Training Engine
6. Hyperparameter Tuning (LR, batch size, dropout, optimizer)
7. Evaluation & Visualisation (Learning curves, ROC, Confusion matrix)
8. Grad-CAM Explainability
9. Comparative Analysis: MLP vs CNN

---

## 📓 Mini-Project 2 — Pretrained CNN + Temporal Modeling

**Notebook:** `notebooks/review02_pretrained_temporal.ipynb`  
**Review:** Review 2

### Objectives
Leverage ImageNet-pretrained CNN backbones for feature extraction and apply recurrent architectures (RNN/LSTM/GRU) to model both spatial and temporal relationships in chest X-ray sequences.

### System Architecture

```
INPUT: Chest X-Ray Images (224 × 224 × 3)
                    |
      +-------------+-------------+
      ▼             ▼             ▼
 ResNet-50     DenseNet-121  EfficientNet-B0
(Pretrained)   (Pretrained)  (Pretrained)
      |             |             |
      ▼             ▼             ▼
Feature Maps (B, C, 7, 7) → Spatial Sequences (B, 49, C)
                    |
       +------------+------------+
       ▼            ▼            ▼
      RNN          LSTM         GRU
   (Vanilla)   (Bidirectional) (Bidirectional)
       |            |            |
       ▼            ▼            ▼
  Bahdanau Attn  Bahdanau Attn  Self-Attention
                    |
              Output: 14 classes
```

### CNN Backbones

| Backbone | Parameters | Feature Dim | Strategy |
|----------|-----------|-------------|----------|
| ResNet-50 | 25.6M | 2048 | layer4 features → 7×7 grid |
| DenseNet-121 | 8.0M | 1024 | Dense features → 7×7 grid |
| EfficientNet-B0 | 5.3M | 1280 | Efficient features → 7×7 grid |

### Model Variants Trained

| # | Model | Backbone | Temporal | Attention | AUC |
|---|-------|----------|----------|-----------|-----|
| 1 | Fine-tuned ResNet-50 | ResNet-50 | — | — | 0.700 |
| 2 | ResNet-50 + RNN | ResNet-50 | Vanilla RNN | Bahdanau | 0.653 |
| 3 | ResNet-50 + LSTM | ResNet-50 | Bidir LSTM | Bahdanau | 0.682 |
| 4 | ResNet-50 + GRU ⭐ | ResNet-50 | Bidir GRU | Self-Attn | **0.710** |
| 5 | DenseNet-121 + LSTM | DenseNet-121 | Bidir LSTM | Bahdanau | ~0.69 |
| 6 | EfficientNet-B0 + GRU | EfficientNet-B0 | Bidir GRU | Self-Attn | ~0.68 |
| 7 | Temporal LSTM | ResNet-50 | LSTM across visits | Temporal Attn | ~0.52 |

**Best model:** ResNet-50 + GRU + Self-Attention → **Mean AUC = 0.710**

### Key Components

| Component | Detail |
|-----------|--------|
| Transfer Learning | ImageNet pretrained; last blocks unfrozen for fine-tuning |
| Temporal Sequences | Grouped by Patient ID, ordered by Follow-up Number (1,471 patients) |
| Training | Mixed Precision (AMP), gradient clipping, early stopping |
| Explainability | Grad-CAM heatmaps on final conv layer |
| Hyperparameter Search | LR × hidden dim × backbone × RNN type |
| Feature Analysis | t-SNE visualisation of learned embeddings |

### Sections
1. Environment Setup & Configuration (seed = 42)
2. Dataset Loading & Preprocessing
3. Pretrained CNN Feature Extractors
4. Fine-Tuning ResNet-50 (Domain Adaptation)
5. Temporal Preprocessing Pipeline (patient visit sequences)
6. Spatial Sequence Models (RNN / LSTM / GRU)
7. True Temporal Models (CNN per frame → RNN across time)
8. Training Engines (standard + temporal)
9. Hyperparameter Search
10. Full Training — All Model Variants
11. Comprehensive Model Comparison
12. Grad-CAM Explainability
13. Attention Weight Visualisation
14. Feature Space Analysis (t-SNE)
15. Final Summary & Model Saving

---

## 📓 Mini-Project 3 — Autoencoder & GAN

**Notebook:** `notebooks/review03_ae_gan.ipynb`  
**Review:** Review 3 — 30th Mar 2026

### Objectives
Learn compact latent representations of chest X-ray images using unsupervised generative models — reconstruct pathological images with high fidelity (AE/VAE) and generate realistic synthetic X-rays (DCGAN).

### Architecture Overview

**Autoencoder (AE)**
```
Encoder: Input(1×64×64)
  → Conv2d(1→32,  4×4, stride=2) + BN + LeakyReLU
  → Conv2d(32→64, 4×4, stride=2) + BN + LeakyReLU
  → Conv2d(64→128,4×4, stride=2) + BN + LeakyReLU
  → Conv2d(128→256,4×4,stride=2) + BN + LeakyReLU
  → Flatten → Linear(4096 → 128)   [Bottleneck: z ∈ R¹²⁸]

Decoder: (symmetric ConvTranspose stack) → Tanh → 1×64×64
Loss: MSE(x̂, x) + λ · PerceptualLoss
```

**Variational Autoencoder (VAE)** *(BONUS)*
```
Encoder → μ ∈ R¹²⁸  and  log σ² ∈ R¹²⁸
z = μ + ε · σ,  ε ~ N(0, 1)    [reparameterisation trick]
Loss: Recon(x̂, x) + β · KL(q(z|x) ∥ p(z))
```

**DCGAN**
```
Generator:  z ∈ R¹²⁸ → Linear(4096) → Reshape(256,4,4)
  → ConvTranspose(256→128) + BN + ReLU  →  8×8
  → ConvTranspose(128→64)  + BN + ReLU  →  16×16
  → ConvTranspose(64→32)   + BN + ReLU  →  32×32
  → ConvTranspose(32→1)    + Tanh       →  64×64

Discriminator: Conv(1→64) → Conv(64→128) → Conv(128→256)
  → Conv(256→512) → Flatten → Linear → Sigmoid
Objective: min_G max_D E[log D(x)] + E[log(1 − D(G(z)))]
```

### Results

| Model | SSIM | MSE | Params |
|-------|------|-----|--------|
| Autoencoder | **0.848** | 0.0078 | 2,431,488 |
| VAE (BONUS) | 0.772 | 0.0180 | 2,955,904 |
| DCGAN Generator | — | — | 1,217,472 |
| DCGAN Discriminator | — | — | 693,633 |

**GAN Training Stability:**

| Epoch | G Loss | D Loss | D(real) | D(fake) | Mode Collapse |
|-------|--------|--------|---------|---------|---------------|
| 10/50 | 1.180 | 0.577 | 0.612 | 0.388 | — |
| 30/50 | 1.093 | 0.573 | 0.607 | 0.393 | — |
| 50/50 | 1.326 | 0.510 | 0.663 | 0.337 | LOW (std=0.334) |

D Loss → 0.51 confirms near Nash equilibrium. Mean pixel std = 0.334 across 100 generated images confirms diversity (no mode collapse).

### Training Configuration

| Setting | Value |
|---------|-------|
| AE / VAE epochs | 30 |
| GAN epochs | 50 |
| Image size | 64 × 64 (grayscale) |
| Latent dim | 128 |
| AE learning rate | 1e-3 |
| GAN Adam β₁ | 0.5 (stability) |
| GAN stability tricks | Label smoothing (0.9/0.1), 2× G updates per D step, grad clip |
| Subset | 10% (10,090 train / 1,122 test) |
| Precision | Mixed (AMP) |

### Sections
1. Dataset — Chest X-Ray (64×64, grayscale)
2. Autoencoder — Encoder-Decoder Symmetric Architecture
3. Variational Autoencoder (BONUS)
4. GAN — Generator & Discriminator
5. Training — Autoencoder (MSE + Perceptual Loss)
6. GAN Training — Min-Max Objective with Stability Tricks
7. Training Dynamics Analysis — Loss Curves
8. Reconstruction Quality — Visual + Quantitative (SSIM)
9. Latent Space Visualization — PCA & t-SNE
10. Latent Space Interpolation (BONUS)
11. Model Saving — All Formats

### Saved Models

| File | Size |
|------|------|
| `ae_weights.pth` | 9.7 MB |
| `vae_weights.pth` | 11.8 MB |
| `gan_generator_weights.pth` | 4.9 MB |
| `models_r3_*.zip` | 24.5 MB |

---

## 📓 Mini-Project 4 — End-to-End Deep Learning System

**Notebook:** `notebooks/review04_densenet_e2e.ipynb`  
**Review:** Review 4 — 30th Mar 2026

### Objectives
Build a complete production-ready DL system for NIH chest X-ray classification: justified architecture selection, structured ablation, Bayesian hyperparameter optimisation, statistical evaluation, and full deployment.

### Architecture Justification

**Why DenseNet-121?**
- Dense connections: each layer receives feature maps from **all** preceding layers → `xl = Hl([x0, x1, ..., xl-1])`
- Feature reuse: low-level edge/texture features propagated to deep classifier — critical for subtle pathologies (Nodule, Infiltration)
- Parameter efficiency: 8.7M vs ResNet-50's 25M at same depth
- **CheXNet (Stanford, 2017):** DenseNet-121 achieved radiologist-level pneumonia detection

**Why EfficientNet-B2?**
- NAS compound scaling: simultaneously scales depth/width/resolution
- Squeeze-and-Excitation (SE) blocks: channel-wise attention — disease-specific feature weighting
- Architecturally diverse from DenseNet → complementary ensemble errors

### Model Performance

| Model | Mean AUC | Mean F1 | Params |
|-------|----------|---------|--------|
| DenseNet-121 (fine-tuned) | 0.7787 | 0.1671 | 2.69M trainable |
| EfficientNet-B2 (fine-tuned) | 0.7775 | — | 2.76M trainable |
| **Ensemble (weighted avg)** ⭐ | **0.7925** | — | — |

### Per-Class AUC (DenseNet-121)

| Disease | AUC | F1 | Precision | Recall |
|---------|-----|-----|-----------|--------|
| Effusion | 0.854 | 0.423 | 0.281 | 0.854 |
| Pneumothorax | 0.809 | 0.222 | 0.132 | 0.711 |
| Edema | 0.821 | 0.126 | 0.067 | 0.810 |
| Consolidation | 0.793 | 0.119 | 0.064 | 0.833 |
| Cardiomegaly | 0.798 | 0.100 | 0.054 | 0.697 |
| Atelectasis | 0.758 | 0.290 | 0.184 | 0.690 |
| Pneumonia | 0.748 | 0.058 | 0.030 | 0.778 |
| Mass | 0.701 | 0.183 | 0.112 | 0.494 |
| Infiltration | 0.669 | 0.339 | 0.231 | 0.640 |
| Nodule | 0.646 | 0.123 | 0.077 | 0.305 |

### Ablation Study

| Condition | Val AUC | Δ vs Full |
|-----------|---------|-----------|
| No pretrain (from scratch) | 0.6427 | −11.3% |
| DenseNet-121 (full model) | 0.7560 | baseline |
| EfficientNet-B2 | 0.7775 | +2.2% |
| No augmentation | 0.7830 | +3.6%* |

*No-augmentation overfits on limited data; generalisation degrades with full training.

### Hyperparameter Optimisation (Optuna)

| Setting | Value |
|---------|-------|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Pruner | MedianPruner |
| Trials | 10 |
| Search space | lr ∈ [1e-4, 1e-3], dropout ∈ [0.2, 0.5], weight_decay ∈ [1e-5, 1e-4] |
| Best lr | 6.75e-4 |
| Best dropout | 0.357 |
| Best weight_decay | 8.29e-5 |
| Best val AUC (2-epoch proxy) | 0.7328 |

### Training Configuration

| Setting | Value |
|---------|-------|
| Loss | BCEWithLogitsLoss (multi-label) |
| Optimiser | AdamW (lr=6.75e-4, wd=8.29e-5) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Precision | Mixed (AMP autocast) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | val AUC (patience=5) |
| Batch size | 32 |
| Best epoch | Ep 9 → AUC = 0.7560 |
| Subset | 15% (13,452 train / 1,682 val / 1,682 test) |

### Deployment

**FastAPI Backend — 7 Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/models` | GET | List available models |
| `/predict` | POST | Single X-ray inference + GradCAM |
| `/predict/batch` | POST | Batch inference |
| `/latent` | POST | AE/VAE latent vector extraction |
| `/temporal` | POST | Patient visit sequence prediction |
| `/report/pdf` | POST | PDF report generation |

**Saved Production Artefacts:**

| File | Type | Size |
|------|------|------|
| `densenet121_traced.pt` | TorchScript | 30.6 MB |
| `efficientnetb2_traced.pt` | TorchScript | 32.1 MB |
| `ensemble_traced.pt` | TorchScript (ensemble) | 63.5 MB |
| `densenet121_model_card.json` | Model card | — |
| `environment.yml` | Conda env | 374 B |
| `requirements.txt` | Pip freeze | 216 B |

### Frontend (React + TypeScript + Vite)

| Page | Description |
|------|-------------|
| `Landing.tsx` | Hero, model showcase, sample X-ray runner, privacy/ethics modals |
| `Upload.tsx` | Drag-and-drop file upload, model multi-select, single/batch routing |
| `Results.tsx` | GradCAM heatmap viewer, per-class predictions, PDF/share |
| `Compare.tsx` | 3-model card grid, ensemble summary |

### Sections
1. Data Engineering — Advanced Pipeline (15% subset, patient-level split)
2. Architecture Justification — DenseNet-121 (theoretical + empirical reasoning)
3. Training with Mixed Precision + Gradient Clipping
4. Experimental Design — Ablation Study
5. Hyperparameter Optimisation — Optuna (BONUS)
6. Performance Evaluation — Statistical Significance
7. Ensemble Model (BONUS)
8. Deployment — FastAPI Ready Model Export (TorchScript + ONNX)
9. Documentation & Reproducibility

---

## ⚙️ Setup & Requirements

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate clinical-curator
pip install -e .
```

### Option 2: pip

```bash
pip install -e .
pip install -r backend/requirements.txt
```

### Option 3: Docker (full stack)

```bash
docker-compose up --build
# Backend:  http://localhost:8000
# Frontend: http://localhost:3000
```

### Dependencies

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pillow
pip install fastapi uvicorn python-multipart
pip install optuna
pip install kagglehub
```

### Hardware

- GPU recommended (CUDA-enabled)
- Mixed precision training (AMP) enabled by default
- All models tested on NVIDIA GPU with ≥ 8 GB VRAM

---

## 🏋️ Training

```bash
# Mini-Project 1: MLP + CNN
python scripts/train_r1.py --data_root /path/to/nih --epochs 20

# Mini-Project 2: Pretrained CNN + RNN temporal models
python scripts/train_r2.py --data_root /path/to/nih --backbone resnet50

# Mini-Project 3: Autoencoder + VAE + DCGAN
python scripts/train_r3.py --data_root /path/to/nih --latent_dim 128

# Mini-Project 4: DenseNet-121 + Optuna HPO + ablation
python scripts/train_r4.py --data_root /path/to/nih --optuna_trials 10
```

---

## 🧪 Testing

```bash
# Full test suite (no GPU or dataset needed)
pytest tests/ -v

# Individual modules
pytest tests/test_models.py      # Forward-pass shape/value, frozen layers
pytest tests/test_trainer.py     # Fit history, loss descent, early stopping
pytest tests/test_api.py         # All 7 endpoints, schema, error codes, PDF
pytest tests/test_gradcam.py     # Heatmap shape/range, overlay, argmax class
```

---

## 🚀 Running the API

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Swagger docs: http://localhost:8000/docs
```

---

## 🔁 Reproducibility

All experiments use **seed = 42** applied to:

```python
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Per-model hyperparameters are stored in `configs/models/*.yaml` for exact reproduction.

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** (primary) | Per-class and macro-averaged; threshold-independent |
| **F1-score** | Optimised threshold per class (0.05–0.10) |
| **SSIM** | Structural Similarity Index (R3: AE/VAE quality) |
| **MSE** | Pixel-level reconstruction error (R3) |
| Visualisations | Learning curves · ROC curves · Confusion matrix · Grad-CAM · t-SNE · Attention maps |

---

## 📋 Results Summary

| Review | Model | Key Result |
|--------|-------|-----------|
| R1 | CNN | AUC ~0.65–0.75 |
| R1 | MLP | AUC ~0.55–0.65 |
| R2 | ResNet-50 + GRU + Self-Attention ⭐ | AUC = **0.710** |
| R2 | Fine-tuned ResNet-50 (no RNN) | AUC = 0.700 |
| R3 | Autoencoder | SSIM = **0.848** |
| R3 | VAE (BONUS) | SSIM = 0.772 |
| R3 | DCGAN | Mode collapse: LOW (std = 0.334) |
| R4 | Ensemble (DenseNet + EfficientNet) ⭐ | AUC = **0.7925** |
| R4 | DenseNet-121 (fine-tuned) | AUC = 0.7787 |
| R4 | EfficientNet-B2 (fine-tuned) | AUC = 0.7775 |

---

## 📚 Course Info

| Field | Detail |
|-------|--------|
| Course | 24AI636 — Deep Learning |
| Mini-Project 1 | Review 1 — 12th Feb 2026 |
| Mini-Project 2 | Review 2 |
| Mini-Project 3 | Review 3 — 30th Mar 2026 |
| Mini-Project 4 | Review 4 — 30th Mar 2026 |
| Dataset | NIH ChestX-ray14 |

---

## 📖 References

- Wang et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*. CVPR.
- Rajpurkar et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. Stanford ML Group.
- Huang et al. (2017). *Densely Connected Convolutional Networks*. CVPR.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
- Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR.
- Goodfellow et al. (2014). *Generative Adversarial Nets*. NeurIPS.
- Kingma & Welling (2014). *Auto-Encoding Variational Bayes*. ICLR.

---

## Contributors

- [@sriguhan7764](https://github.com/sriguhan7764)

## License

MIT
