# Architectural ML & DL from Scratch

A micro-library of clean, decoupled, and from-scratch implementations covering classical machine learning, modern sparse large language models, and generative frameworks.

## Repository Architecture

grad-zero/
├── .github/                      
│
├── classical_ml/                 # ── DOMAIN 1: PURE NUMPY ──
│   ├── supervised/               # Regression, SVMs, Trees, and Ensembles
│   └── unsupervised/             # Clustering (K-Means, DBSCAN), PCA, GMMs
│
├── pytorch_impl/                 # ── DOMAIN 2: DEEP LEARNING ──
│   ├── modules/                  # MHA, MoE, RoPE, KV-Cache, Flash Attention
│   └── architectures/            
│       ├── vision/               # CNNs, ResNet, ViT, Swin Transformers
│       └── sequence/             # LSTMs, Vanilla Transformers, Deepseek styled MoE, Mamba
│
├── alignment_and_robustness/     # ── DOMAIN 3: ALIGNMENT & SECURITY ──
│   ├── distillation/             # SLM Teacher-Student logit matching
│   └── alignment/                # DPO, GRPO, and PPO implementations
│ 
│
└── generative_models/            # ── DOMAIN 4: GENERATIVE FRAMEWORKS ──
    ├── diffusion/                # DDPM, DDIM, Score-based SDEs
    ├── autoencoders/             # Standard VAEs and VQ-VAEs
    └── adversarial/              # Vanilla GANs, DCGANs, Normalizing Flows
