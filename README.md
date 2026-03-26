# NIH Chest X-Ray Multi-Label Classification
### 24AI636 Deep Learning — Mini-Projects 1 & 2

> Multi-label thoracic disease classification from the NIH ChestX-ray14 dataset using MLP, CNN, Pretrained Backbones, and Temporal Sequence Models.

---

## 📁 Project Structure

| File | Project | Description |
|------|---------|-------------|
| `NIH_ChestXRay_MLP_CNN_Classification_MiniProject1.ipynb` | Mini-Project 1 | Baseline MLP & CNN classifiers |
| `NIH_ChestXRay_Pretrained_CNN_Temporal_Modeling_MiniProject2.ipynb` | Mini-Project 2 | Pretrained CNNs + RNN/LSTM/GRU temporal models |

---

## 📊 Dataset

- **Name**: NIH ChestX-ray14
- **Size**: 112,120 frontal-view chest X-ray images
- **Labels**: 14 thoracic disease classes (multi-label)
- **Split**: Patient-level 80% train / 10% val / 10% test (prevents data leakage)
- **Input resolution**: 224 × 224 × 3
- **Source**: Downloaded via `kagglehub`

### Disease Classes
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

---

## 📓 Mini-Project 1 — MLP & CNN Classification

**File**: `NIH_ChestXRay_MLP_CNN_Classification_MiniProject1.ipynb`  
**Review**: Review 1 — 12th Feb 2026

### Objectives
Implement and compare baseline deep learning models for multi-label chest X-ray classification, covering the full ML pipeline from preprocessing to explainability.

### Architecture

**MLP**
```
Input (150,528) → Dense(1024) → BN → ReLU → Dropout
               → Dense(512)  → BN → ReLU → Dropout
               → Dense(128)  → Dense(14) → Sigmoid
```

**CNN**
```
Input (3×224×224)
→ Conv Block 1: Conv2d(3→32)   → BN → ReLU → MaxPool(2×2)
→ Conv Block 2: Conv2d(32→64)  → BN → ReLU → MaxPool(2×2)
→ Conv Block 3: Conv2d(64→128) → BN → ReLU → MaxPool(2×2)
→ Conv Block 4: Conv2d(128→256)→ BN → ReLU → MaxPool(2×2)
→ FC(1024) → Dropout → FC(14) → Sigmoid
```

### Key Components

| Component | Choice | Reason |
|-----------|--------|--------|
| Loss | `BCEWithLogitsLoss` + `pos_weight` | Numerically stable; handles class imbalance |
| Optimizer | AdamW + weight decay | Adaptive LR with L2 regularisation |
| Scheduler | `ReduceLROnPlateau` | Adaptive LR based on val AUC |
| Augmentation | RandomHorizontalFlip + RandomRotation(10°) | Train only |

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

### MLP vs CNN Summary

| Aspect | MLP | CNN |
|--------|-----|-----|
| Spatial awareness | ❌ Flattens image | ✅ Preserves local structure |
| Parameter efficiency | Low | High |
| Expected AUC | ~0.55–0.65 | ~0.65–0.75 |
| Training stability | May oscillate | Smoother convergence |

---

## 📓 Mini-Project 2 — Pretrained CNN + Temporal Modeling

**File**: `NIH_ChestXRay_Pretrained_CNN_Temporal_Modeling_MiniProject2.ipynb`  
**Review**: Review 2

### Objectives
Leverage ImageNet-pretrained CNN backbones for feature extraction and apply recurrent architectures (RNN/LSTM/GRU) to model both spatial and temporal relationships in chest X-ray sequences.

### System Architecture

```
INPUT: Chest X-Ray Images (224 × 224 × 3)
                    |
      +-------------+-------------+
      v             v             v
 ResNet-50     DenseNet-121  EfficientNet-B0
(Pretrained)   (Pretrained)  (Pretrained)
      |             |             |
      v             v             v
Feature Maps (B, C, 7, 7) → Spatial Sequences (B, 49, C)
                    |
       +------------+------------+
       v            v            v
      RNN          LSTM         GRU
   (Vanilla)   (Bidirectional) (Bidirectional)
       |            |            |
       v            v            v
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

| # | Model | Backbone | Temporal | Attention |
|---|-------|----------|----------|-----------|
| 1 | Fine-tuned ResNet-50 (no RNN) | ResNet-50 | — | — |
| 2 | ResNet-50 + RNN (spatial) | ResNet-50 | Vanilla RNN | Bahdanau |
| 3 | ResNet-50 + LSTM (spatial) | ResNet-50 | Bidirectional LSTM | Bahdanau |
| 4 | ResNet-50 + GRU (spatial) | ResNet-50 | Bidirectional GRU | Self-Attention |
| 5 | DenseNet-121 + LSTM (spatial) | DenseNet-121 | Bidirectional LSTM | Bahdanau |
| 6 | EfficientNet-B0 + GRU (spatial) | EfficientNet-B0 | Bidirectional GRU | Self-Attention |
| 7 | Temporal LSTM (real-time visits) | ResNet-50 | LSTM across visits | Temporal Attn |

### Key Components

| Component | Detail |
|-----------|--------|
| Transfer Learning | ImageNet pretrained; last blocks unfrozen for fine-tuning |
| Temporal Sequences | Grouped by Patient ID, ordered by Follow-up Number |
| Training | Mixed Precision (AMP), gradient clipping, early stopping |
| Explainability | Grad-CAM heatmaps on final conv layer |
| Hyperparameter Search | LR × hidden dim × backbone × RNN type |
| Feature Analysis | t-SNE visualisation of learned embeddings |

### Sections
1. Environment Setup & Configuration
2. Configuration & Reproducibility (seed = 42)
3. Dataset Loading & Preprocessing
4. Pretrained CNN Feature Extractors
5. Fine-Tuning ResNet-50 (Domain Adaptation)
6. Temporal Preprocessing Pipeline (patient visit sequences)
7. Spatial Sequence Models (RNN / LSTM / GRU)
8. True Temporal Models (CNN per frame → RNN across time)
9. Training Engines (standard + temporal)
10. Evaluation & Visualisation Toolkit
11. Hyperparameter Search
12. Full Training — All Model Variants
13. Comprehensive Model Comparison
14. Grad-CAM Explainability
15. Attention Weight Visualisation
16. Feature Space Analysis (t-SNE)
17. Final Summary & Model Saving

---

## ⚙️ Setup & Requirements

### Dependencies

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pillow
pip install kagglehub
```

### Hardware
- GPU recommended (CUDA-enabled)
- Mixed precision training (AMP) enabled by default

### Reproducibility
All experiments use `seed = 42` for:
```python
random, os.environ['PYTHONHASHSEED'], numpy, torch, torch.cuda
```

---

## 📈 Evaluation Metrics

- **Primary**: ROC-AUC (per-class and macro-averaged)
- **Secondary**: Accuracy at 0.5 threshold
- **Visualisations**: Learning curves, ROC curves, Confusion matrix, Grad-CAM, t-SNE, Attention weights

---

## 📚 Course Info

| Field | Detail |
|-------|--------|
| Course | 24AI636 — Deep Learning |
| Mini-Project 1 | Review 1 — 12th Feb 2026 |
| Mini-Project 2 | Review 2 |
| Dataset | NIH ChestX-ray14 |
